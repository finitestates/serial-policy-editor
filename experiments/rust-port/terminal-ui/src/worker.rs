//! Framed RPC client for the persistent Python inference worker.

use std::collections::BTreeMap;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

pub const PROTOCOL_VERSION: u64 = 1;
const MAX_JSON_FRAME_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Deserialize)]
struct Response {
    version: u64,
    request_id: u64,
    op: String,
    ok: bool,
    error: Option<String>,
    result: Option<Value>,
    backend_wall_s: f64,
    binary_dtype: Option<String>,
    binary_count: usize,
    binary_bytes: usize,
    source_dtype: Option<String>,
}

#[derive(Clone, Debug)]
struct RpcResult {
    result: Value,
    binary: Vec<u8>,
}

#[derive(Clone, Debug, Default, Serialize)]
struct OperationMetric {
    calls: u64,
    backend_wall_s: f64,
    round_trip_wall_s: f64,
    outside_backend_wall_s: f64,
    request_bytes: u64,
    response_bytes: u64,
    logit_payload_bytes: u64,
    logit_source_dtypes: Vec<String>,
}

pub struct BackendWorker {
    child: Child,
    input: ChildStdin,
    output: ChildStdout,
    next_request_id: u64,
    vocabulary_size: usize,
    hello: Value,
    metrics: BTreeMap<String, OperationMetric>,
    capture_dir: Option<PathBuf>,
    metrics_path: Option<PathBuf>,
    closed: bool,
}

impl BackendWorker {
    pub fn spawn_from_env() -> io::Result<Self> {
        let python = required_path("SPE_REAL_MODEL_PYTHON")?;
        let script = required_path("SPE_REAL_MODEL_WORKER")?;
        let profile = required_path("SPE_REAL_MODEL_PROFILE")?;
        let mut command = Command::new(python);
        command
            .arg(script)
            .arg("--profile")
            .arg(profile)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped());
        if let Some(model_root) = env::var_os("SPE_REAL_MODEL_ROOT") {
            command.arg("--model-root").arg(model_root);
        }
        if let Some(path) = env::var_os("SPE_REAL_MODEL_WORKER_STDERR") {
            let stderr = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(path)?;
            command.stderr(Stdio::from(stderr));
        } else {
            command.stderr(Stdio::inherit());
        }
        let mut child = command.spawn()?;
        let input = child
            .stdin
            .take()
            .ok_or_else(|| io::Error::other("Python worker did not provide a stdin pipe"))?;
        let output = child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("Python worker did not provide a stdout pipe"))?;
        let mut worker = Self {
            child,
            input,
            output,
            next_request_id: 1,
            vocabulary_size: 0,
            hello: Value::Null,
            metrics: BTreeMap::new(),
            capture_dir: env::var_os("SPE_REAL_MODEL_CAPTURE_DIR").map(PathBuf::from),
            metrics_path: env::var_os("SPE_REAL_MODEL_METRICS_PATH").map(PathBuf::from),
            closed: false,
        };
        let started = Instant::now();
        let (response, response_bytes) = read_response(&mut worker.output)?;
        if response.version != PROTOCOL_VERSION
            || response.request_id != 0
            || response.op != "hello"
            || response.binary_bytes != 0
        {
            return Err(invalid_data("worker sent an invalid startup response"));
        }
        if !response.ok {
            return Err(invalid_data(format!(
                "Python inference worker startup failed: {}",
                response.error.as_deref().unwrap_or("unspecified error")
            )));
        }
        let hello = response
            .result
            .ok_or_else(|| invalid_data("worker startup response omitted metadata"))?;
        let vocabulary_size = hello
            .get("vocabulary_size")
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| invalid_data("worker reported an invalid vocabulary size"))?;
        if let Some(path) = env::var_os("SPE_REAL_MODEL_METADATA_PATH") {
            let path = PathBuf::from(path);
            if let Some(parent) = path.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(
                &path,
                serde_json::to_vec_pretty(&hello).map_err(invalid_data)?,
            )?;
        }
        let metric = OperationMetric {
            calls: 1,
            backend_wall_s: response.backend_wall_s,
            round_trip_wall_s: started.elapsed().as_secs_f64(),
            outside_backend_wall_s: 0.0,
            request_bytes: 0,
            response_bytes: u64::try_from(response_bytes).unwrap_or(u64::MAX),
            logit_payload_bytes: 0,
            logit_source_dtypes: Vec::new(),
        };
        let mut metric = metric;
        metric.outside_backend_wall_s = (metric.round_trip_wall_s - metric.backend_wall_s).max(0.0);
        worker.metrics.insert("startup".to_owned(), metric);
        worker.vocabulary_size = vocabulary_size;
        worker.hello = hello;
        Ok(worker)
    }

    pub fn hello(&self) -> &Value {
        &self.hello
    }

    pub fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    pub fn tokenize(&mut self, text: &str) -> io::Result<Value> {
        Ok(self
            .rpc(
                "tokenize",
                json!({"text": text, "add_bos": true, "special": true}),
            )?
            .result)
    }

    pub fn reset(&mut self, token_ids: &[i64]) -> io::Result<()> {
        self.rpc("reset", json!({"token_ids": token_ids}))?;
        Ok(())
    }

    pub fn eval(&mut self, token_ids: &[i64]) -> io::Result<()> {
        self.rpc("eval", json!({"token_ids": token_ids}))?;
        Ok(())
    }

    pub fn logits(&mut self, boundary: u64) -> io::Result<Vec<f64>> {
        let response = self.rpc("logits", json!({"boundary": boundary}))?;
        let values = decode_logits(&response.result, &response.binary, self.vocabulary_size)?;
        if let Some(directory) = &self.capture_dir {
            fs::create_dir_all(directory)?;
            let path = directory.join(format!("live-boundary-{boundary}.f64le"));
            let mut file = File::create(path)?;
            file.write_all(&response.binary)?;
            file.sync_all()?;
        }
        Ok(values)
    }

    pub fn render(&mut self, token_ids: &[i64]) -> io::Result<String> {
        self.rpc("render", json!({"token_ids": token_ids}))?
            .result
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| invalid_data("worker render response omitted text"))
    }

    pub fn token_text(&mut self, token_id: i64) -> io::Result<String> {
        self.rpc("token_text", json!({"token_id": token_id}))?
            .result
            .get("text")
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| invalid_data("worker token_text response omitted text"))
    }

    pub fn is_eog(&mut self, token_id: i64) -> io::Result<bool> {
        self.rpc("is_eog", json!({"token_id": token_id}))?
            .result
            .get("is_eog")
            .and_then(Value::as_bool)
            .ok_or_else(|| invalid_data("worker is_eog response omitted result"))
    }

    pub fn eog_token_ids(&mut self) -> io::Result<Vec<i64>> {
        self.rpc("eog_token_ids", json!({}))?
            .result
            .get("token_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid_data("worker eog_token_ids response omitted token IDs"))?
            .iter()
            .map(|value| {
                value
                    .as_i64()
                    .ok_or_else(|| invalid_data("worker returned a non-integer EOG token ID"))
            })
            .collect()
    }

    pub fn shutdown(&mut self) -> io::Result<()> {
        if self.closed {
            return Ok(());
        }
        let response = self.rpc("shutdown", json!({}))?;
        if response.result.get("closed").and_then(Value::as_bool) != Some(true) {
            return Err(invalid_data("worker did not confirm shutdown"));
        }
        let status = self.child.wait()?;
        if !status.success() {
            return Err(invalid_data(format!("Python worker exited with {status}")));
        }
        self.closed = true;
        self.write_metrics()
    }

    fn rpc(&mut self, op: &str, mut request: Value) -> io::Result<RpcResult> {
        let request_id = self.next_request_id;
        self.next_request_id = self
            .next_request_id
            .checked_add(1)
            .ok_or_else(|| invalid_data("worker request ID overflow"))?;
        let object = request
            .as_object_mut()
            .ok_or_else(|| invalid_data("worker request must be a JSON object"))?;
        object.insert("version".to_owned(), json!(PROTOCOL_VERSION));
        object.insert("request_id".to_owned(), json!(request_id));
        object.insert("op".to_owned(), json!(op));
        let started = Instant::now();
        let request_bytes = write_json_frame(&mut self.input, &request)?;
        let (response, response_header_bytes) = read_response(&mut self.output)?;
        if response.version != PROTOCOL_VERSION
            || response.request_id != request_id
            || response.op != op
        {
            return Err(invalid_data(format!(
                "worker response does not match request {request_id} {op:?}"
            )));
        }
        let expected_bytes = match response.binary_count.checked_mul(8) {
            Some(value) => value,
            None => return Err(invalid_data("worker binary element count overflow")),
        };
        if expected_bytes != response.binary_bytes {
            return Err(invalid_data(format!(
                "worker declared {} payload bytes for {} elements",
                response.binary_bytes, response.binary_count
            )));
        }
        if response.binary_bytes > self.vocabulary_size.saturating_mul(8) {
            return Err(invalid_data(
                "worker binary payload exceeds vocabulary limit",
            ));
        }
        let mut binary = vec![0; response.binary_bytes];
        if !binary.is_empty() {
            read_exact(&mut self.output, &mut binary)?;
        }
        if !response.ok {
            return Err(invalid_data(format!(
                "Python inference worker operation {op:?} failed: {}",
                response.error.as_deref().unwrap_or("unspecified error")
            )));
        }
        let result = response
            .result
            .ok_or_else(|| invalid_data("worker response omitted result"))?;
        if op == "logits"
            && (response.binary_dtype.as_deref() != Some("f64le")
                || response.binary_count != self.vocabulary_size)
        {
            return Err(invalid_data(format!(
                "worker logits metadata must be f64le with {} values",
                self.vocabulary_size
            )));
        }
        if op != "logits" && response.binary_bytes != 0 {
            return Err(invalid_data("worker sent an unexpected binary payload"));
        }
        let round_trip_wall_s = started.elapsed().as_secs_f64();
        let metric = self.metrics.entry(op.to_owned()).or_default();
        metric.calls += 1;
        metric.backend_wall_s += response.backend_wall_s;
        metric.round_trip_wall_s += round_trip_wall_s;
        metric.outside_backend_wall_s += (round_trip_wall_s - response.backend_wall_s).max(0.0);
        metric.request_bytes += u64::try_from(request_bytes).unwrap_or(u64::MAX);
        metric.response_bytes +=
            u64::try_from(response_header_bytes + binary.len()).unwrap_or(u64::MAX);
        metric.logit_payload_bytes += u64::try_from(binary.len()).unwrap_or(u64::MAX);
        if op == "logits"
            && let Some(source_dtype) = response.source_dtype
            && !metric.logit_source_dtypes.contains(&source_dtype)
        {
            metric.logit_source_dtypes.push(source_dtype);
        }
        Ok(RpcResult { result, binary })
    }

    fn write_metrics(&self) -> io::Result<()> {
        let Some(path) = &self.metrics_path else {
            return Ok(());
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let total = self
            .metrics
            .values()
            .fold(OperationMetric::default(), |mut sum, item| {
                sum.calls += item.calls;
                sum.backend_wall_s += item.backend_wall_s;
                sum.round_trip_wall_s += item.round_trip_wall_s;
                sum.outside_backend_wall_s += item.outside_backend_wall_s;
                sum.request_bytes += item.request_bytes;
                sum.response_bytes += item.response_bytes;
                sum.logit_payload_bytes += item.logit_payload_bytes;
                for source_dtype in &item.logit_source_dtypes {
                    if !sum.logit_source_dtypes.contains(source_dtype) {
                        sum.logit_source_dtypes.push(source_dtype.clone());
                    }
                }
                sum
            });
        let payload = json!({
            "protocol_version": PROTOCOL_VERSION,
            "operations": self.metrics,
            "totals": {
                "backend_wall_s": total.backend_wall_s,
                "round_trip_wall_s": total.round_trip_wall_s,
                "outside_backend_wall_s": total.outside_backend_wall_s,
                "request_bytes": total.request_bytes,
                "response_bytes": total.response_bytes,
                "total_bytes": total.request_bytes + total.response_bytes,
                "logit_payload_bytes": total.logit_payload_bytes,
                "logit_dtype": "f64le",
                "source_dtypes": total.logit_source_dtypes
            }
        });
        fs::write(
            path,
            serde_json::to_vec_pretty(&payload).map_err(invalid_data)?,
        )
    }
}

impl Drop for BackendWorker {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

fn required_path(name: &str) -> io::Result<PathBuf> {
    env::var_os(name)
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, format!("{name} is required")))
}

fn read_response(reader: &mut impl Read) -> io::Result<(Response, usize)> {
    let mut prefix = [0_u8; 4];
    read_exact(reader, &mut prefix)?;
    let length = usize::try_from(u32::from_le_bytes(prefix))
        .map_err(|_| invalid_data("worker JSON frame length overflow"))?;
    if length == 0 || length > MAX_JSON_FRAME_BYTES {
        return Err(invalid_data(format!(
            "invalid worker JSON frame length {length}"
        )));
    }
    let mut payload = vec![0; length];
    read_exact(reader, &mut payload)?;
    let response = serde_json::from_slice(&payload).map_err(invalid_data)?;
    Ok((response, length + 4))
}

fn write_json_frame(writer: &mut impl Write, value: &Value) -> io::Result<usize> {
    let payload = serde_json::to_vec(value).map_err(invalid_data)?;
    if payload.is_empty() || payload.len() > MAX_JSON_FRAME_BYTES {
        return Err(invalid_data(format!(
            "invalid worker JSON frame length {}",
            payload.len()
        )));
    }
    let length = u32::try_from(payload.len())
        .map_err(|_| invalid_data("worker JSON frame length overflow"))?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(&payload)?;
    writer.flush()?;
    Ok(payload.len() + 4)
}

fn read_exact(reader: &mut impl Read, output: &mut [u8]) -> io::Result<()> {
    reader.read_exact(output).map_err(|error| {
        if error.kind() == io::ErrorKind::UnexpectedEof {
            invalid_data("truncated worker protocol frame")
        } else {
            error
        }
    })
}

fn decode_logits(result: &Value, bytes: &[u8], vocabulary_size: usize) -> io::Result<Vec<f64>> {
    let declared_count = result
        .get("shape")
        .and_then(Value::as_array)
        .filter(|shape| shape.len() == 1)
        .and_then(|shape| shape[0].as_u64())
        .and_then(|value| usize::try_from(value).ok())
        .ok_or_else(|| invalid_data("worker logits shape must be one-dimensional"))?;
    if declared_count != vocabulary_size || bytes.len() != vocabulary_size.saturating_mul(8) {
        return Err(invalid_data(format!(
            "worker logits payload mismatch: expected {vocabulary_size} f64 values"
        )));
    }
    let (chunks, remainder) = bytes.as_chunks::<8>();
    if !remainder.is_empty() {
        return Err(invalid_data(
            "worker logits payload has a partial f64 value",
        ));
    }
    let values = chunks
        .iter()
        .map(|chunk| f64::from_le_bytes(*chunk))
        .collect::<Vec<_>>();
    if values.iter().any(|value| !value.is_finite()) {
        return Err(invalid_data("worker logits contain a non-finite value"));
    }
    Ok(values)
}

fn invalid_data(error: impl std::fmt::Display) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, error.to_string())
}

#[cfg(test)]
mod tests {
    use super::{decode_logits, read_response, write_json_frame};
    use serde_json::{Value, json};
    use std::io::{Cursor, ErrorKind};

    #[test]
    fn json_frames_round_trip_and_reject_truncation() {
        let mut bytes = Vec::new();
        write_json_frame(
            &mut bytes,
            &json!({
                "version": 1, "request_id": 4, "op": "eval", "ok": true,
                "error": null, "result": {}, "backend_wall_s": 0.0,
                "binary_dtype": null, "binary_count": 0, "binary_bytes": 0
            }),
        )
        .unwrap();
        let (response, size) = read_response(&mut Cursor::new(bytes.clone())).unwrap();
        assert_eq!(response.op, "eval");
        assert_eq!(size, bytes.len());

        let error = read_response(&mut Cursor::new(bytes[..bytes.len() - 1].to_vec())).unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidData);
    }

    #[test]
    fn logits_require_one_finite_value_per_vocabulary_item() {
        let encoded = [1.25_f64.to_le_bytes(), (-2.5_f64).to_le_bytes()].concat();
        assert_eq!(
            decode_logits(&json!({"shape": [2]}), &encoded, 2).unwrap(),
            [1.25, -2.5]
        );
        assert!(decode_logits(&json!({"shape": [1, 2]}), &encoded, 2).is_err());
        assert!(decode_logits(&json!({"shape": [1]}), &encoded, 2).is_err());

        let invalid = [f64::INFINITY.to_le_bytes()].concat();
        assert!(decode_logits(&json!({"shape": [1]}), &invalid, 1).is_err());
        assert!(decode_logits(&json!({"shape": [1]}), &[0; 7], 1).is_err());
    }

    #[test]
    fn worker_error_header_is_a_json_object() {
        let value: Value = json!({
            "version": 1,
            "request_id": 8,
            "op": "eval",
            "ok": false,
            "error": "backend exploded",
            "result": null,
            "backend_wall_s": 0.0,
            "binary_dtype": null,
            "binary_count": 0,
            "binary_bytes": 0
        });
        let mut bytes = Vec::new();
        write_json_frame(&mut bytes, &value).unwrap();
        let (response, _) = read_response(&mut Cursor::new(bytes)).unwrap();
        assert!(!response.ok);
        assert_eq!(response.error.as_deref(), Some("backend exploded"));
    }
}
