import importlib
from pathlib import Path

import numpy as np
import pytest


def _load_mvdec(monkeypatch):
    tensorflow = pytest.importorskip("tensorflow")
    representation_dir = (
        Path(__file__).resolve().parents[1] / "src" / "representation_learning"
    )
    monkeypatch.syspath_prepend(str(representation_dir))
    return tensorflow, importlib.import_module("MVDEC_dense")


def test_independent_view2_protocol_is_a_bounded_cross_paper_diagnostic(
    monkeypatch,
):
    _, mvdec = _load_mvdec(monkeypatch)
    protocol = mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL
    contract = protocol.manifest_contract(0.001)

    assert protocol.claim_scope == "controlled_diagnostic_not_exact_paper_truth"
    assert protocol.view1_refinement_policy == mvdec.VIEW1_REFINEMENT_FROZEN_DEKM_F
    assert protocol.refinement_representation == (mvdec.REFINEMENT_REPRESENTATION_VIEW2)
    assert protocol.view2_architecture_id == mvdec.VIEW2_ARCHITECTURE_ID
    assert protocol.reconstruction_weight == 0.0
    assert protocol.greedy_eigen_direction == "largest"
    assert protocol.greedy_target_mode == "frozen_snapshot"
    assert protocol.refinement_batch_size_policy == mvdec.FULL_DATASET_BATCH_POLICY
    assert contract["schedule"]["refinement"]["representation"] == (
        mvdec.REFINEMENT_REPRESENTATION_VIEW2
    )
    assert contract["schedule"]["refinement"]["fusion_timing"] == (
        "final_evaluation_only"
    )
    assert not contract["objective"]["loss_terms"]["L1_reconstruction"]["optimized"]


def test_independent_view2_refinement_targets_ignore_view1(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    view2_output = np.array([[1.0, 2.0, 30.0], [3.0, 4.0, 40.0]])
    first_view1 = np.zeros_like(view2_output)
    second_view1 = np.full_like(view2_output, 999.0)

    first = mvdec.refinement_embedding(
        first_view1,
        view2_output,
        mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL,
    )
    second = mvdec.refinement_embedding(
        second_view1,
        view2_output,
        mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL,
    )

    np.testing.assert_array_equal(first, view2_output[:, :2])
    np.testing.assert_array_equal(second, first)


def test_independent_view2_refinement_excludes_reconstruction_head(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "input_shape", 3)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    monkeypatch.setattr(mvdec, "view2_base_units", 4)
    model2 = mvdec.model_view2(load_weights=False)
    model1 = type("FrozenView1", (), {"trainable_variables": []})()
    variables = mvdec.refinement_trainable_variables(
        model1,
        model2,
        mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL,
    )
    reconstruction_variables = model2.get_layer(
        "view2_reconstruction"
    ).trainable_variables

    assert variables
    assert not {id(variable) for variable in variables} & {
        id(variable) for variable in reconstruction_variables
    }

    x = tensorflow.ones((2, 3))
    with tensorflow.GradientTape() as tape:
        view2_output = model2(x)
        latent = mvdec.refinement_embedding(
            tensorflow.zeros_like(view2_output),
            view2_output,
            mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL,
        )
        loss = tensorflow.reduce_sum(latent**2)
    gradients = tape.gradient(loss, variables)

    assert all(gradient is not None for gradient in gradients)


def test_independent_view2_optimizer_step_does_not_change_view1(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "input_shape", 3)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    monkeypatch.setattr(mvdec, "view1_filters", [4, 4, 6])
    monkeypatch.setattr(mvdec, "view2_base_units", 4)
    model1 = mvdec.model_view1(load_weights=False)
    model2 = mvdec.model_view2(load_weights=False)
    mvdec.configure_view1_refinement(model1, mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL)
    before = [weight.numpy().copy() for weight in model1.weights]
    variables = mvdec.refinement_trainable_variables(
        model1,
        model2,
        mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL,
    )
    assert not {id(variable) for variable in model1.variables} & {
        id(variable) for variable in variables
    }

    x = tensorflow.ones((2, 3))
    target = tensorflow.zeros((2, 2))
    optimizer = tensorflow.keras.optimizers.Adam(learning_rate=0.001)
    with tensorflow.GradientTape() as tape:
        view1_output = model1(x)
        view2_output = model2(x)
        predicted = mvdec.refinement_embedding(
            view1_output,
            view2_output,
            mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL,
        )
        loss = mvdec.independent_dekm_greedy_loss(target, predicted)
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(mvdec.validate_refinement_gradients(gradients, variables))

    for expected, actual in zip(before, model1.weights, strict=True):
        np.testing.assert_array_equal(actual.numpy(), expected)


def test_independent_dekm_loss_matches_release_mse(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    y_true = tensorflow.zeros((2, 2))
    y_pred = tensorflow.constant([[1.0, 3.0], [2.0, 4.0]])

    loss = mvdec.independent_dekm_greedy_loss(y_true, y_pred)

    np.testing.assert_allclose(loss.numpy(), [5.0, 10.0])


def test_calibrated_independent_view2_protocol_is_preregistered(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    protocol = mvdec.CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL
    contract = protocol.manifest_contract(0.001)
    refinement = contract["schedule"]["refinement"]

    assert protocol.protocol_id == mvdec.CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL_ID
    assert protocol.claim_scope == "controlled_diagnostic_not_exact_paper_truth"
    assert protocol.view1_refinement_policy == mvdec.VIEW1_REFINEMENT_FROZEN_DEKM_F
    assert protocol.refinement_representation == mvdec.REFINEMENT_REPRESENTATION_FUSED
    assert protocol.view2_architecture_id == mvdec.VIEW2_ARCHITECTURE_ID
    assert protocol.reconstruction_weight == 0.0
    assert protocol.kmeans_weight == 0.0
    assert protocol.scatter_trace_weight == 0.0
    assert protocol.greedy_weight == 1.0
    assert protocol.greedy_eigen_direction == "largest"
    assert protocol.greedy_target_mode == "frozen_snapshot"
    assert protocol.refinement_batch_size_policy == mvdec.FULL_DATASET_BATCH_POLICY
    assert refinement["representation"] == mvdec.REFINEMENT_REPRESENTATION_FUSED
    assert refinement["fusion_timing"] == "fixed_calibration_before_refinement"
    assert refinement["update_interval"] == 10
    assert refinement["max_training_steps"] == 14_000
    assert refinement["calibration_policy"] == {
        "method": mvdec.CENTERED_RMS_CALIBRATION_METHOD,
        "source_phase": mvdec.CENTERED_RMS_CALIBRATION_SOURCE_PHASE,
        "label_free": True,
        "calibration_count": 1,
        "recomputed_during_refinement": False,
        "claim_scope": mvdec.CENTERED_RMS_CALIBRATION_CLAIM_SCOPE,
    }


def test_calibrated_refinement_uses_fixed_scale_after_view2_update(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    view1_output = tensorflow.constant([[1.0, 3.0, 9.0], [5.0, 7.0, 9.0]])
    view2_before = tensorflow.constant([[2.0, 4.0, 8.0], [6.0, 10.0, 8.0]])
    calibration = mvdec.CenteredRmsCalibration.fit(
        view1_output.numpy()[:, :2],
        view2_before.numpy()[:, :2],
    )
    scale_before = calibration.view2_scale_factor
    metadata_before = calibration.to_metadata()
    view2_after = view2_before + tensorflow.constant(
        [[10.0, -2.0, 0.0], [-3.0, 12.0, 0.0]]
    )

    refined = mvdec.refinement_embedding(
        view1_output,
        view2_after,
        mvdec.CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL,
        calibration,
    )
    expected = (view1_output[:, :2] + scale_before * view2_after[:, :2]) / 2

    np.testing.assert_array_equal(refined.numpy(), expected.numpy())
    assert calibration.view2_scale_factor == scale_before
    assert calibration.to_metadata() == metadata_before


def test_calibrated_refinement_trains_only_view2_representation_path(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "input_shape", 3)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    monkeypatch.setattr(mvdec, "view1_filters", [4, 4, 6])
    monkeypatch.setattr(mvdec, "view2_base_units", 4)
    model1 = mvdec.model_view1(load_weights=False)
    model2 = mvdec.model_view2(load_weights=False)
    variables = mvdec.refinement_trainable_variables(
        model1,
        model2,
        mvdec.CALIBRATED_INDEPENDENT_DEKM_VIEW2_PROTOCOL,
    )
    variable_ids = {id(variable) for variable in variables}
    reconstruction_ids = {
        id(variable)
        for variable in model2.get_layer("view2_reconstruction").trainable_variables
    }

    assert variable_ids
    assert not variable_ids & {id(variable) for variable in model1.variables}
    assert not variable_ids & reconstruction_ids


def test_frozen_view1_integrity_proof_is_bitwise_and_tamper_evident(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    from pipeline.mvdec_integrity import validate_frozen_view1_proof

    monkeypatch.setattr(mvdec, "hidden_units", 2)
    weights = (np.array([[1.0, 2.0]]), np.array([3.0]))
    output = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    proof = mvdec.frozen_view1_integrity_proof(
        weights,
        tuple(weight.copy() for weight in weights),
        output,
        output.copy(),
        np.arange(output.shape[0]),
    )

    validated = validate_frozen_view1_proof(proof, output[:, :2])

    assert validated == proof
    assert proof["weights_bitwise_equal"] is True
    assert proof["output_bitwise_equal"] is True
    assert proof["latent_bitwise_equal"] is True

    for field, value in (
        ("weights_bitwise_equal", False),
        ("weight_count", 0),
        ("h_view1_sha256", "0" * 64),
        ("output_before_sha256", "not-a-hash"),
    ):
        tampered = dict(proof)
        tampered[field] = value
        with pytest.raises(ValueError):
            validate_frozen_view1_proof(tampered, output[:, :2])


def test_frozen_view1_integrity_proof_rejects_changed_view1(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    monkeypatch.setattr(mvdec, "hidden_units", 2)
    weights = (np.array([[1.0, 2.0]]),)
    changed_weights = (np.array([[1.0, 2.1]]),)
    output = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])

    with pytest.raises(ValueError, match="changed"):
        mvdec.frozen_view1_integrity_proof(
            weights,
            changed_weights,
            output,
            output.copy(),
            None,
        )


def test_imported_joint_view1_protocol_pins_both_policies(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    protocol = mvdec.IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL
    contract = protocol.manifest_contract(0.001)

    assert protocol.view1_initialization_policy == (
        mvdec.VIEW1_INITIALIZATION_IMPORTED_DEKM_F
    )
    assert protocol.view1_refinement_policy == mvdec.VIEW1_REFINEMENT_FROZEN_DEKM_F
    assert protocol.view2_architecture_id == mvdec.VIEW2_ARCHITECTURE_ID
    assert contract["architecture"]["view2"] == mvdec.VIEW2_ARCHITECTURE_ID
    assert contract["schedule"]["pretraining"]["view1_initialization_policy"] == (
        mvdec.VIEW1_INITIALIZATION_IMPORTED_DEKM_F
    )
    assert contract["schedule"]["refinement"]["view1_refinement_policy"] == (
        mvdec.VIEW1_REFINEMENT_FROZEN_DEKM_F
    )
    assert contract["schedule"]["final_evaluation"]["view1_semantics"] == (
        "frozen_imported_dekm_f_checkpoint_unchanged_during_refinement"
    )


def test_imported_joint_view1_resolver_and_run_gate(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    imported_joint = mvdec.IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL
    ordinary = mvdec.PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL
    manifest = Path("manifest.json")

    assert mvdec.resolve_view1_manifest_argument(imported_joint, manifest) == manifest
    with pytest.raises(ValueError, match="required"):
        mvdec.resolve_view1_manifest_argument(imported_joint, None)
    with pytest.raises(ValueError, match="only"):
        mvdec.resolve_view1_manifest_argument(ordinary, manifest)
    mvdec.validate_imported_view1_run_gate(imported_joint, "REUTERS", 42, 1)
    for dataset, seed, runs in (
        ("RCV1", 42, 1),
        ("REUTERS", 43, 1),
        ("REUTERS", 42, 2),
    ):
        with pytest.raises(ValueError, match="REUTERS seed 42"):
            mvdec.validate_imported_view1_run_gate(
                imported_joint,
                dataset,
                seed,
                runs,
            )


def test_imported_joint_refinement_optimizes_both_views(monkeypatch):
    tf, mvdec = _load_mvdec(monkeypatch)
    model1 = tf.keras.Sequential([tf.keras.layers.Dense(2)])
    model2 = tf.keras.Sequential([tf.keras.layers.Dense(2)])
    sample = tf.ones((2, 2))
    model1(sample)
    model2(sample)
    protocol = mvdec.IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL

    mvdec.configure_view1_refinement(model1, protocol)
    variables = mvdec.refinement_trainable_variables(model1, model2, protocol)
    variable_ids = {id(variable) for variable in variables}

    assert model1.trainable is False
    assert not model1.trainable_variables
    assert {id(variable) for variable in model2.trainable_variables} <= variable_ids
    assert not ({id(variable) for variable in model1.weights} & variable_ids)


def test_view1_refinement_transition_is_tamper_evident(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    from pipeline.mvdec_integrity import validate_view1_refinement_transition

    monkeypatch.setattr(mvdec, "hidden_units", 2)
    monkeypatch.setattr(mvdec, "n_clusters", 2)
    initial_weights = (np.array([[1.0, 2.0]]), np.array([0.0]))
    final_weights = (np.array([[1.1, 2.0]]), np.array([0.1]))
    initial_output = np.array(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [5.0, 5.0, 1.0], [5.1, 5.0, 1.0]]
    )
    final_output = initial_output.copy()
    final_output[:, 0] += np.array([0.0, 0.05, 0.1, 0.15])
    labels = np.array([0, 0, 1, 1])
    proof = mvdec.view1_refinement_transition_proof(
        initial_weights,
        final_weights,
        initial_output,
        final_output,
        labels,
        42,
        None,
        0.25,
    )
    final_latent = final_output[:, :2]

    assert validate_view1_refinement_transition(proof, final_latent) == proof
    assert proof["view1_nonzero_gradient_observed"] is True
    assert proof["weights_changed"] is True
    assert proof["latents_changed"] is True

    tampered_cases = (
        {key: value for key, value in proof.items() if key != "weight_count"},
        {**proof, "view1_max_abs_gradient": 0.0},
        {**proof, "view1_nonzero_gradient_observed": False},
        {
            **proof,
            "final_weights_sha256": proof["initial_weights_sha256"],
        },
        {
            **proof,
            "final_latent_sha256": proof["initial_latent_sha256"],
        },
        {**proof, "h_view1_sha256": "0" * 64},
    )
    for tampered in tampered_cases:
        with pytest.raises(ValueError):
            validate_view1_refinement_transition(tampered, final_latent)

    altered_latent = final_latent.copy()
    altered_latent[0, 0] += 1.0
    with pytest.raises(ValueError, match="latent hash"):
        validate_view1_refinement_transition(proof, altered_latent)


def test_min_refinement_protocol_only_adds_a_preregistered_step_floor(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    base = mvdec.IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL
    protocol = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL
    base_contract = base.manifest_contract(0.001)
    contract = protocol.manifest_contract(0.001)

    assert protocol.claim_scope == "controlled_diagnostic_not_exact_paper_truth"
    assert protocol.min_training_steps == 500
    assert base.min_training_steps == 0
    assert "min_training_steps" not in base_contract["stopping"]
    assert contract["stopping"]["min_training_steps"] == 500
    for field in (
        "view1_refinement_policy",
        "view2_architecture_id",
        "greedy_eigen_direction",
        "greedy_target_mode",
        "fusion_contract",
        "max_training_steps",
        "refinement_batch_size_policy",
    ):
        assert getattr(protocol, field) == getattr(base, field)
    mvdec.validate_imported_view1_run_gate(protocol, "REUTERS", 42, 1)
    with pytest.raises(ValueError, match="REUTERS seed 42"):
        mvdec.validate_imported_view1_run_gate(protocol, "REUTERS", 43, 1)


@pytest.mark.parametrize("bad_floor", [-1, True, 14_001, 1.5])
def test_min_training_steps_must_be_a_bounded_non_negative_int(
    monkeypatch,
    bad_floor,
):
    from dataclasses import replace

    _, mvdec = _load_mvdec(monkeypatch)
    with pytest.raises(ValueError, match="min_training_steps"):
        replace(mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL, min_training_steps=bad_floor)


def test_refinement_stop_waits_for_the_minimum_step_floor(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    base = mvdec.IMPORTED_DEKM_F_VIEW1_JOINT_EQ11_RELEASE_L2_PROTOCOL
    floored = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL

    assert mvdec.refinement_converged(base, 10, 0, 10_000, 0.001)
    assert not mvdec.refinement_converged(floored, 10, 0, 10_000, 0.001)
    assert not mvdec.refinement_converged(floored, 500, 11, 10_000, 0.001)
    assert mvdec.refinement_converged(floored, 500, 10, 10_000, 0.001)


def test_every_registered_protocol_uses_loader_accepted_provenance_labels(
    monkeypatch,
):
    from pipeline.mvdec_contract import DECISION_PROVENANCE_CLASSIFICATIONS

    _, mvdec = _load_mvdec(monkeypatch)
    for protocol in mvdec.PUBLIC_PROTOCOLS.values():
        labels = {category for _, category, _ in protocol.decision_provenance}
        assert labels <= DECISION_PROVENANCE_CLASSIFICATIONS, protocol.protocol_id


def test_min_refinement_joint_protocol_differs_only_by_trainable_view1(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    frozen = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL
    joint = mvdec.MIN_REFINEMENT_JOINT_VIEW1_PROTOCOL

    assert joint.view1_refinement_policy == mvdec.VIEW1_REFINEMENT_JOINT
    assert joint.view1_initialization_policy == (
        mvdec.VIEW1_INITIALIZATION_IMPORTED_DEKM_F
    )
    assert mvdec.is_imported_joint_view1(joint)
    assert not mvdec.is_imported_joint_view1(frozen)
    assert joint.min_training_steps == frozen.min_training_steps == 500
    for field in (
        "view2_architecture_id",
        "greedy_eigen_direction",
        "greedy_target_mode",
        "fusion_contract",
        "max_training_steps",
        "refinement_batch_size_policy",
    ):
        assert getattr(joint, field) == getattr(frozen, field)
    provenance = {key: category for key, category, _ in joint.decision_provenance}
    assert provenance["view1_refinement_policy"] == "paper_explicit"
    mvdec.validate_imported_view1_run_gate(joint, "REUTERS", 42, 1)
