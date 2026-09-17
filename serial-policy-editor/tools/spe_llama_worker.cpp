// Narrow llama.cpp compatibility worker for SPE hidden-state capture.
//
// This is intentionally a one-shot tool, not an inference server.  It links
// against a known llama.cpp build, calls the native layer-input capture hooks
// directly, and emits a JSON response that the Python vector workbench can
// turn into a portable SPE artifact.

#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// These hooks are implemented by SPE's pinned llama.cpp checkout.  They are
// currently C++-linkage functions because the checkout has not promoted them
// into the public C ABI.  Calling them here is deliberate: the compatibility
// boundary is now this worker's build, rather than Python resolving compiler-
// generated symbols from an arbitrary shared library at runtime.
void llama_set_embeddings_layer_inp(
    llama_context * ctx, uint32_t layer, bool enabled);
float * llama_get_embeddings_layer_inp(
    llama_context * ctx, uint32_t layer);

namespace {

struct Arguments {
    std::string model;
    std::string prompt_a;
    std::string prompt_b;
    std::string prompt_a_file;
    std::string prompt_b_file;
    std::string output;
    int layer_start = 0;
    int layer_end = 0;
    std::string position = "last";
    uint32_t n_ctx = 4096;
    uint32_t n_batch = 4096;
    int32_t n_threads = 0;
    int32_t n_gpu_layers = 0;
    bool normalize = true;
};

struct Capture {
    int token_count = 0;
    std::vector<float> values;
};

static void fail(const std::string & message) {
    throw std::runtime_error(message);
}

static int parse_int(const std::string & value, const char * name) {
    size_t consumed = 0;
    int result = 0;
    try {
        result = std::stoi(value, &consumed);
    } catch (const std::exception &) {
        fail(std::string(name) + " must be an integer");
    }
    if (consumed != value.size()) {
        fail(std::string(name) + " must be an integer");
    }
    return result;
}

static uint32_t parse_uint(const std::string & value, const char * name) {
    const int result = parse_int(value, name);
    if (result < 1) {
        fail(std::string(name) + " must be positive");
    }
    return static_cast<uint32_t>(result);
}

static std::string read_text_file(const std::string & path, const char * label) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        fail(std::string("could not read ") + label + " file: " + path);
    }
    std::ostringstream contents;
    contents << file.rdbuf();
    return contents.str();
}

static std::string json_string(const std::string & value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char ch : value) {
        switch (ch) {
            case '"': result << "\\\""; break;
            case '\\': result << "\\\\"; break;
            case '\b': result << "\\b"; break;
            case '\f': result << "\\f"; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (ch < 0x20) {
                    result << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0') << static_cast<int>(ch)
                           << std::dec << std::setfill(' ');
                } else {
                    result << static_cast<char>(ch);
                }
        }
    }
    result << '"';
    return result.str();
}

static void print_help(const char * program) {
    std::cout
        << "usage: " << program << " --model MODEL"
        << " (--prompt-a TEXT | --prompt-a-file PATH)"
        << " (--prompt-b TEXT | --prompt-b-file PATH)"
        << " (--layer N | --layer-range START END)"
        << " [options]\n\n"
        << "Capture llama.cpp residual states for a prompt pair and emit JSON.\n"
        << "Prompt A minus Prompt B is returned for each selected layer.\n\n"
        << "options:\n"
        << "  --model PATH\n"
        << "  --prompt-a TEXT | --prompt-a-file PATH\n"
        << "  --prompt-b TEXT | --prompt-b-file PATH\n"
        << "  --layer N | --layer-range START END\n"
        << "  --position first|last       (default: last)\n"
        << "  --no-normalize              keep raw differences\n"
        << "  --n-ctx N                   (default: 4096)\n"
        << "  --n-batch N                 (default: 4096)\n"
        << "  --n-threads N               (default: llama.cpp default)\n"
        << "  --n-gpu-layers N            (default: 0)\n"
        << "  --output PATH               (default: stdout)\n";
}

static Arguments parse_arguments(int argc, char ** argv) {
    Arguments args;
    for (int i = 1; i < argc; ++i) {
        const std::string option = argv[i];
        auto require_value = [&](const char * name) -> std::string {
            if (i + 1 >= argc) {
                fail(std::string(name) + " requires a value");
            }
            return argv[++i];
        };
        if (option == "--help" || option == "-h") {
            print_help(argv[0]);
            std::exit(0);
        } else if (option == "--model") {
            args.model = require_value("--model");
        } else if (option == "--prompt-a") {
            args.prompt_a = require_value("--prompt-a");
        } else if (option == "--prompt-b") {
            args.prompt_b = require_value("--prompt-b");
        } else if (option == "--prompt-a-file") {
            args.prompt_a_file = require_value("--prompt-a-file");
        } else if (option == "--prompt-b-file") {
            args.prompt_b_file = require_value("--prompt-b-file");
        } else if (option == "--layer") {
            args.layer_start = parse_int(require_value("--layer"), "--layer");
            args.layer_end = args.layer_start;
        } else if (option == "--layer-range") {
            args.layer_start = parse_int(require_value("--layer-range"), "--layer-start");
            args.layer_end = parse_int(require_value("--layer-range"), "--layer-end");
        } else if (option == "--position") {
            args.position = require_value("--position");
        } else if (option == "--no-normalize") {
            args.normalize = false;
        } else if (option == "--n-ctx") {
            args.n_ctx = parse_uint(require_value("--n-ctx"), "--n-ctx");
        } else if (option == "--n-batch") {
            args.n_batch = parse_uint(require_value("--n-batch"), "--n-batch");
        } else if (option == "--n-threads") {
            args.n_threads = parse_int(require_value("--n-threads"), "--n-threads");
        } else if (option == "--n-gpu-layers") {
            args.n_gpu_layers = parse_int(require_value("--n-gpu-layers"), "--n-gpu-layers");
        } else if (option == "--output") {
            args.output = require_value("--output");
        } else {
            fail("unknown option: " + option);
        }
    }

    if (args.model.empty()) {
        fail("--model is required");
    }
    if (!args.prompt_a.empty() && !args.prompt_a_file.empty()) {
        fail("use only one of --prompt-a and --prompt-a-file");
    }
    if (!args.prompt_b.empty() && !args.prompt_b_file.empty()) {
        fail("use only one of --prompt-b and --prompt-b-file");
    }
    if (args.prompt_a.empty() && args.prompt_a_file.empty()) {
        fail("one of --prompt-a or --prompt-a-file is required");
    }
    if (args.prompt_b.empty() && args.prompt_b_file.empty()) {
        fail("one of --prompt-b or --prompt-b-file is required");
    }
    if (args.layer_start < 1 || args.layer_end < args.layer_start) {
        fail("a positive --layer or an ascending --layer-range is required");
    }
    if (args.position != "first" && args.position != "last") {
        fail("--position must be first or last");
    }
    if (args.n_threads < 0 || args.n_gpu_layers < 0) {
        fail("--n-threads and --n-gpu-layers cannot be negative");
    }
    if (args.n_batch > args.n_ctx) {
        args.n_batch = args.n_ctx;
    }
    return args;
}

static std::vector<llama_token> tokenize(
    const llama_vocab * vocab, const std::string & text) {
    const int32_t required = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, true, true);
    if (required == std::numeric_limits<int32_t>::min() || required >= 0) {
        fail("llama.cpp returned an invalid tokenization size");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(-required));
    const int32_t actual = llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
        static_cast<int32_t>(tokens.size()), true, true);
    if (actual < 0 || actual != static_cast<int32_t>(tokens.size())) {
        fail("llama.cpp failed to tokenize prompt");
    }
    return tokens;
}

static std::vector<Capture> capture_prompt(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::string & prompt,
    int layer_start,
    int layer_end,
    const std::string & position,
    int width,
    uint32_t n_ctx,
    uint32_t n_batch) {
    if (prompt.empty()) {
        fail("prompts must be nonempty");
    }
    const std::vector<llama_token> tokens = tokenize(vocab, prompt);
    if (tokens.empty()) {
        fail("prompt produced no tokens");
    }
    if (tokens.size() > n_ctx || tokens.size() > n_batch) {
        fail("prompt exceeds the worker context or batch size");
    }

    llama_memory_clear(llama_get_memory(ctx), true);
    const llama_batch batch = llama_batch_get_one(
        const_cast<llama_token *>(tokens.data()), static_cast<int32_t>(tokens.size()));
    const int result = llama_decode(ctx, batch);
    if (result != 0) {
        fail("llama.cpp failed to evaluate prompt (error " + std::to_string(result) + ")");
    }

    const size_t row = position == "first" ? 0 : tokens.size() - 1;
    std::vector<Capture> captures;
    for (int layer = layer_start; layer <= layer_end; ++layer) {
        float * pointer = llama_get_embeddings_layer_inp(
            ctx, static_cast<uint32_t>(layer));
        if (pointer == nullptr) {
            fail("llama.cpp returned no hidden-state capture for layer " +
                 std::to_string(layer));
        }
        Capture capture;
        capture.token_count = static_cast<int>(tokens.size());
        capture.values.assign(pointer + row * static_cast<size_t>(width),
                              pointer + (row + 1) * static_cast<size_t>(width));
        if (std::any_of(capture.values.begin(), capture.values.end(),
                        [](float value) { return !std::isfinite(value); })) {
            fail("llama.cpp returned non-finite hidden-state values");
        }
        captures.push_back(std::move(capture));
    }
    return captures;
}

static double norm(const std::vector<float> & values) {
    long double sum = 0.0;
    for (float value : values) {
        sum += static_cast<long double>(value) * static_cast<long double>(value);
    }
    return std::sqrt(static_cast<double>(sum));
}

static std::string architecture(const llama_model * model) {
    char value[128] = {};
    if (llama_model_meta_val_str(model, "general.architecture", value, sizeof(value)) < 0) {
        return "unknown";
    }
    return value;
}

static void write_vector(std::ostream & out, const std::vector<float> & values) {
    out << '[';
    out << std::setprecision(9);
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        out << values[i];
    }
    out << ']';
}

static void write_json(
    std::ostream & out,
    const Arguments & args,
    const llama_model * model,
    int width,
    int layer_count,
    const std::vector<Capture> & captures_a,
    const std::vector<Capture> & captures_b,
    const std::vector<std::vector<float>> & directions,
    const std::vector<double> & raw_norms) {
    out << "{\n"
        << "  \"protocol\": \"spe-llama-worker-v1\",\n"
        << "  \"operation\": \"hidden-state-pair\",\n"
        << "  \"backend\": {\n"
        << "    \"name\": \"llama.cpp\",\n"
        << "    \"version\": " << json_string(llama_version()) << ",\n"
        << "    \"capture_api\": \"layer-input-native-worker\"\n"
        << "  },\n"
        << "  \"model\": {\n"
        << "    \"filename\": "
        << json_string(std::filesystem::path(args.model).filename().string()) << ",\n"
        << "    \"model_type\": " << json_string(architecture(model)) << ",\n"
        << "    \"vocabulary_size\": "
        << llama_vocab_n_tokens(llama_model_get_vocab(model)) << ",\n"
        << "    \"hidden_state_width\": " << width << ",\n"
        << "    \"hidden_state_layer_count\": " << layer_count << "\n"
        << "  },\n"
        << "  \"target\": {\n"
        << "    \"site\": \"decoder-block-output-residual\",\n"
        << "    \"layer_numbering\": \"one-based\",\n"
        << "    \"layer_start\": " << args.layer_start << ",\n"
        << "    \"layer_end\": " << args.layer_end << ",\n"
        << "    \"position\": " << json_string(args.position) << "\n"
        << "  },\n"
        << "  \"prompts\": {\n"
        << "    \"a_token_count\": " << captures_a.front().token_count << ",\n"
        << "    \"b_token_count\": " << captures_b.front().token_count << "\n"
        << "  },\n"
        << "  \"normalized\": " << (args.normalize ? "true" : "false") << ",\n"
        << "  \"directions\": {\n";
    for (size_t index = 0; index < directions.size(); ++index) {
        if (index != 0) {
            out << ",\n";
        }
        const int layer = args.layer_start + static_cast<int>(index);
        out << "    " << json_string(std::to_string(layer)) << ": ";
        write_vector(out, directions[index]);
    }
    out << "\n  },\n  \"raw_delta_norms\": {\n";
    for (size_t index = 0; index < raw_norms.size(); ++index) {
        if (index != 0) {
            out << ",\n";
        }
        const int layer = args.layer_start + static_cast<int>(index);
        out << "    " << json_string(std::to_string(layer)) << ": "
            << std::setprecision(17) << raw_norms[index];
    }
    out << "\n  }\n}\n";
}

} // namespace

int main(int argc, char ** argv) {
    try {
        const Arguments args = parse_arguments(argc, argv);
        const std::string prompt_a = args.prompt_a.empty()
            ? read_text_file(args.prompt_a_file, "prompt A") : args.prompt_a;
        const std::string prompt_b = args.prompt_b.empty()
            ? read_text_file(args.prompt_b_file, "prompt B") : args.prompt_b;

        llama_backend_init();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = args.n_gpu_layers;
        llama_model * model = llama_model_load_from_file(args.model.c_str(), model_params);
        if (model == nullptr) {
            llama_backend_free();
            fail("llama.cpp could not load model: " + args.model);
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = args.n_ctx;
        context_params.n_batch = args.n_batch;
        context_params.n_ubatch = std::min(context_params.n_ubatch, args.n_batch);
        context_params.embeddings = true;
        if (args.n_threads > 0) {
            context_params.n_threads = args.n_threads;
            context_params.n_threads_batch = args.n_threads;
        }
        llama_context * ctx = llama_init_from_model(model, context_params);
        if (ctx == nullptr) {
            llama_model_free(model);
            llama_backend_free();
            fail("llama.cpp could not create a context");
        }

        const int width = llama_model_n_embd(model);
        const int model_layers = llama_model_n_layer(model);
        const int capture_layers = std::max(1, model_layers - 1);
        if (width < 1 || model_layers < 1) {
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            fail("llama.cpp reported invalid model dimensions");
        }
        if (args.layer_end > capture_layers) {
            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            fail("layer range exceeds llama.cpp control-vector layers (1.." +
                 std::to_string(capture_layers) + ")");
        }

        for (int layer = args.layer_start; layer <= args.layer_end; ++layer) {
            llama_set_embeddings_layer_inp(ctx, static_cast<uint32_t>(layer), true);
        }
        const llama_vocab * vocab = llama_model_get_vocab(model);
        std::vector<Capture> captures_a;
        std::vector<Capture> captures_b;
        std::vector<std::vector<float>> directions;
        std::vector<double> raw_norms;
        captures_a = capture_prompt(
            ctx, vocab, prompt_a, args.layer_start, args.layer_end,
            args.position, width, args.n_ctx, args.n_batch);
        captures_b = capture_prompt(
            ctx, vocab, prompt_b, args.layer_start, args.layer_end,
            args.position, width, args.n_ctx, args.n_batch);
        for (size_t index = 0; index < captures_a.size(); ++index) {
            std::vector<float> direction(width);
            for (int i = 0; i < width; ++i) {
                direction[i] = captures_a[index].values[i] - captures_b[index].values[i];
            }
            const double raw_norm = norm(direction);
            if (args.normalize && raw_norm > 0.0) {
                for (float & value : direction) {
                    value = static_cast<float>(value / raw_norm);
                }
            }
            directions.push_back(std::move(direction));
            raw_norms.push_back(raw_norm);
        }

        std::ofstream file;
        std::ostream * output = &std::cout;
        if (!args.output.empty()) {
            file.open(args.output);
            if (!file) {
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                fail("could not open output: " + args.output);
            }
            output = &file;
        }
        write_json(*output, args, model, width, capture_layers,
                   captures_a, captures_b, directions, raw_norms);

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "spe-llama-worker: error: " << error.what() << '\n';
        return 2;
    }
}
