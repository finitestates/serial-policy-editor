use _native::execute_request;
use serde_json::Value;

#[test]
fn python_generated_history_cases_match_the_typed_kernel() {
    let fixture: Value =
        serde_json::from_str(include_str!("../fixtures/history-cases.json")).unwrap();
    let cases = fixture["cases"].as_array().unwrap();

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let expected = &case["expected"];
        let actual = execute_request(&case["request"]);
        if expected["ok"] == true {
            assert_eq!(&actual, expected, "fixture case {name}");
        } else {
            assert_eq!(
                actual["ok"], false,
                "fixture case {name} unexpectedly succeeded: {actual}"
            );
            assert_eq!(
                actual["error"]["kind"], expected["error"]["kind"],
                "fixture case {name}: {actual}"
            );
            if name == "action-unknown-kind"
                || name == "unknown-action-in-history"
                || name == "malformed-known-action-in-history"
            {
                assert_eq!(
                    actual["error"]["message"], expected["error"]["message"],
                    "action parsing in a history must keep the Python diagnostic"
                );
            }
        }
    }
}
