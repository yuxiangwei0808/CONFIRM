from bench.run_confirm_layer import compute_layer_result


def test_confirm_layer_filters_neuroclaw_false_confirms():
    result = compute_layer_result(
        {
            "model": "fixture",
            "per_claim": [
                {
                    "claim_id": "pos_keep",
                    "label_class": "known_positive",
                    "neuroclaw_decision": "CONFIRMS",
                    "confirm_final_label": "confirmed",
                },
                {
                    "claim_id": "pos_no_effect",
                    "label_class": "known_positive",
                    "neuroclaw_decision": "AMBIGUOUS",
                    "confirm_final_label": "confirmed",
                },
                {
                    "claim_id": "null_filtered",
                    "label_class": "known_null",
                    "neuroclaw_decision": "CONFIRMS",
                    "confirm_final_label": "fragile",
                },
                {
                    "claim_id": "fragile_no_effect",
                    "label_class": "fragile",
                    "neuroclaw_decision": "NO-EFFECT",
                    "confirm_final_label": "fragile",
                },
            ],
        }
    )

    assert result["neuroclaw_alone_FCR"]["count"] == 1
    assert result["neuroclaw_alone_FCR"]["denominator"] == 2
    assert result["neuroclaw_confirm_layer_FCR"]["count"] == 0
    assert result["neuroclaw_confirm_layer_FCR"]["denominator"] == 2
    assert result["neuroclaw_alone_TPR"]["count"] == 1
    assert result["neuroclaw_confirm_layer_TPR"]["count"] == 1
    assert result["neuroclaw_false_confirms_converted_to_abstentions"] == ["null_filtered"]
    assert "ci95_exact" in result["neuroclaw_confirm_layer_FCR"]
