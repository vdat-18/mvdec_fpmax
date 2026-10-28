import importlib
from pathlib import Path

import numpy as np
import pytest
from pipeline import data


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


def test_min_refinement_scratch_protocol_trains_both_views_with_true_bottleneck(
    monkeypatch,
):
    _, mvdec = _load_mvdec(monkeypatch)
    scratch = mvdec.MIN_REFINEMENT_SCRATCH_PROTOCOL
    base = mvdec.PAPER_RESOLVED_RELEASE_L2_FULL_BATCH_PROTOCOL

    assert scratch.min_training_steps == 500
    assert scratch.view2_architecture_id == mvdec.VIEW2_ARCHITECTURE_ID
    assert scratch.protocol_id in mvdec.PUBLIC_PROTOCOLS
    for field in (
        "view1_initialization_policy",
        "greedy_eigen_direction",
        "greedy_target_mode",
        "refinement_batch_size_policy",
        "input_normalization",
        "fusion_contract",
    ):
        assert getattr(scratch, field) == getattr(base, field)


def test_min_refinement_wide_view2_protocol_changes_only_the_stem_width(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    frozen = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL
    wide = mvdec.MIN_REFINEMENT_WIDE_VIEW2_PROTOCOL

    assert wide.view2_base_units == 256
    assert frozen.view2_base_units == 0
    assert "view2_base_units" not in frozen.manifest_contract(0.001)["architecture"]
    assert wide.manifest_contract(0.001)["architecture"]["view2_base_units"] == 256
    for field in (
        "min_training_steps",
        "view2_architecture_id",
        "view1_refinement_policy",
        "fusion_contract",
        "scatter_trace_weight",
    ):
        assert getattr(wide, field) == getattr(frozen, field)
    with pytest.raises(ValueError, match="view2_base_units"):
        mvdec.replace(wide, view2_base_units=-1)


def test_min_refinement_aligned_protocol_adds_only_a_preregistered_alignment_term(
    monkeypatch,
):
    _, mvdec = _load_mvdec(monkeypatch)
    frozen = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL
    aligned = mvdec.MIN_REFINEMENT_ALIGNED_VIEW1_PROTOCOL
    contract = aligned.manifest_contract(0.001)
    terms = contract["objective"]["loss_terms"]

    assert aligned.view1_refinement_policy == mvdec.VIEW1_REFINEMENT_FROZEN_DEKM_F
    assert aligned.min_training_steps == frozen.min_training_steps == 500
    assert aligned.alignment_weight == 1.0
    assert aligned.alignment_mode == mvdec.ALIGNMENT_FULL_BATCH_CENTERED_ROW_L2
    assert aligned.scatter_trace_weight == 0.0
    assert terms["L5_cross_view_alignment"]["optimized"]
    assert not terms["L3_scatter_trace"]["optimized"]
    assert contract["objective"]["loss_scale_policy"] == "sum_squared_dimensions"
    assert aligned.view2_architecture_id == frozen.view2_architecture_id
    assert aligned.refinement_batch_size_policy == mvdec.FULL_DATASET_BATCH_POLICY
    provenance = {key: category for key, category, _ in aligned.decision_provenance}
    assert provenance["cross_view_alignment"] == "ablation"
    mvdec.validate_imported_view1_run_gate(aligned, "REUTERS", 42, 1)



def test_scratch_concat_protocol_only_changes_fusion(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    concat = mvdec.MIN_REFINEMENT_SCRATCH_CONCAT_PROTOCOL
    scratch = mvdec.MIN_REFINEMENT_SCRATCH_PROTOCOL

    assert concat.fusion_contract == mvdec.FUSION_ENCODER_CONCATENATION
    assert concat.protocol_id in mvdec.PUBLIC_PROTOCOLS
    assert concat.protocol_id in mvdec.PROTOCOL_IDS
    for field in (
        "min_training_steps",
        "view2_architecture_id",
        "view1_initialization_policy",
        "greedy_eigen_direction",
        "greedy_target_mode",
        "refinement_batch_size_policy",
        "input_normalization",
        "loss_scale_policy",
    ):
        assert getattr(concat, field) == getattr(scratch, field)
    contract = concat.manifest_contract(0.001)
    assert contract["architecture"]["fusion"] == mvdec.FUSION_ENCODER_CONCATENATION


def test_concat_fusion_matches_in_numpy_and_tensorflow(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    rng = np.random.default_rng(0)
    width = mvdec.hidden_units
    view1 = rng.normal(size=(5, width + 3)).astype(np.float32)
    view2 = rng.normal(size=(5, width + 3)).astype(np.float32)

    fused_np = mvdec.fused_latent_embedding(
        view1, view2, mvdec.FUSION_ENCODER_CONCATENATION
    )
    fused_tf = mvdec.fused_latent_embedding(
        tensorflow.constant(view1),
        tensorflow.constant(view2),
        mvdec.FUSION_ENCODER_CONCATENATION,
    ).numpy()

    assert fused_np.shape == (5, 2 * width)
    np.testing.assert_array_equal(fused_np[:, :width], view1[:, :width])
    np.testing.assert_array_equal(fused_np[:, width:], view2[:, :width])
    np.testing.assert_array_equal(fused_np, fused_tf)


def test_independent_view2_seed_sweep_keeps_seed42_view1(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    sweep = mvdec.INDEPENDENT_DEKM_VIEW2_SEED_SWEEP_PROTOCOL
    base = mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL

    assert sweep.protocol_id in mvdec.PUBLIC_PROTOCOLS
    assert sweep.protocol_id in mvdec.INDEPENDENT_DEKM_VIEW2_PROTOCOL_IDS
    for field in (
        "view1_refinement_policy",
        "refinement_representation",
        "refinement_objective",
        "greedy_eigen_direction",
        "greedy_target_mode",
        "fusion_contract",
        "view2_architecture_id",
    ):
        assert getattr(sweep, field) == getattr(base, field)
    mvdec.validate_imported_view1_run_gate(sweep, "REUTERS", 57, 1)
    with pytest.raises(ValueError):
        mvdec.validate_imported_view1_run_gate(sweep, "RCV1", 57, 1)
    with pytest.raises(ValueError):
        mvdec.validate_imported_view1_run_gate(base, "REUTERS", 57, 1)


def test_scratch_l2_concat_and_paper_rms_protocols_change_one_field(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    scratch = mvdec.MIN_REFINEMENT_SCRATCH_PROTOCOL
    l2_concat = mvdec.MIN_REFINEMENT_SCRATCH_L2_CONCAT_PROTOCOL
    paper_rms = mvdec.MIN_REFINEMENT_SCRATCH_PAPER_RMS_PROTOCOL

    assert l2_concat.fusion_contract == mvdec.FUSION_ENCODER_L2_NORMALIZED_CONCATENATION
    assert l2_concat.input_normalization == scratch.input_normalization
    assert paper_rms.input_normalization == mvdec.PAPER_RMS_NORMALIZATION
    assert paper_rms.fusion_contract == scratch.fusion_contract
    for protocol in (l2_concat, paper_rms):
        assert protocol.protocol_id in mvdec.PUBLIC_PROTOCOLS
        assert protocol.protocol_id in mvdec.PROTOCOL_IDS
        for field in (
            "min_training_steps",
            "view2_architecture_id",
            "view1_initialization_policy",
            "greedy_eigen_direction",
            "greedy_target_mode",
            "refinement_batch_size_policy",
            "loss_scale_policy",
        ):
            assert getattr(protocol, field) == getattr(scratch, field)


def test_l2_concat_fusion_balances_views_in_numpy_and_tensorflow(monkeypatch):
    tensorflow, mvdec = _load_mvdec(monkeypatch)
    rng = np.random.default_rng(0)
    width = mvdec.hidden_units
    view1 = 10 * rng.normal(size=(5, width + 3)).astype(np.float32)
    view2 = rng.normal(size=(5, width + 3)).astype(np.float32)

    fused_np = mvdec.fused_latent_embedding(
        view1, view2, mvdec.FUSION_ENCODER_L2_NORMALIZED_CONCATENATION
    )
    fused_tf = mvdec.fused_latent_embedding(
        tensorflow.constant(view1),
        tensorflow.constant(view2),
        mvdec.FUSION_ENCODER_L2_NORMALIZED_CONCATENATION,
    ).numpy()

    assert fused_np.shape == (5, 2 * width)
    for block in (fused_np[:, :width], fused_np[:, width:]):
        np.testing.assert_allclose(np.linalg.norm(block, axis=1), 1, rtol=1e-6)
    np.testing.assert_allclose(fused_np, fused_tf, rtol=1e-6)


def test_table1_subset_protocol_only_changes_sample_counts(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    scratch = mvdec.MIN_REFINEMENT_SCRATCH_PROTOCOL
    subset = mvdec.MIN_REFINEMENT_SCRATCH_TABLE1_SUBSET_PROTOCOL

    assert subset.table1_sample_counts and not scratch.table1_sample_counts
    assert subset.protocol_id in mvdec.PUBLIC_PROTOCOLS
    assert mvdec.MVDEC_TABLE1_SAMPLE_COUNTS["REUTERS"] == 210
    for field in ("fusion_contract", "input_normalization", "min_training_steps"):
        assert getattr(subset, field) == getattr(scratch, field)
    contract = subset.manifest_contract(0.001)
    assert contract["preprocessing"]["sample_counts"]["REUTERS"] == 210
    assert "sample_counts" not in scratch.manifest_contract(0.001)["preprocessing"]


def test_minibatch_and_no_l1_protocols_change_one_field(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    scratch = mvdec.MIN_REFINEMENT_SCRATCH_PROTOCOL
    minibatch = mvdec.MIN_REFINEMENT_SCRATCH_MINIBATCH_PROTOCOL
    no_l1 = mvdec.MIN_REFINEMENT_SCRATCH_NO_L1_PROTOCOL

    assert minibatch.refinement_batch_size_policy == mvdec.FIXED_256_BATCH_POLICY
    assert minibatch.reconstruction_weight == scratch.reconstruction_weight
    assert no_l1.reconstruction_weight == 0.0
    assert no_l1.refinement_batch_size_policy == scratch.refinement_batch_size_policy
    for protocol in (minibatch, no_l1):
        assert protocol.protocol_id in mvdec.PUBLIC_PROTOCOLS
        for field in ("fusion_contract", "input_normalization", "min_training_steps"):
            assert getattr(protocol, field) == getattr(scratch, field)
    l1_tenth = mvdec.MIN_REFINEMENT_SCRATCH_L1_TENTH_PROTOCOL
    assert l1_tenth.reconstruction_weight == 0.1
    assert l1_tenth.kmeans_weight == scratch.kmeans_weight


def test_frozen_view1_l2_concat_sweeps_view2_seed_with_seed42_view1(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    frozen = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL
    sweep = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_L2_CONCAT_PROTOCOL

    assert sweep.fusion_contract == mvdec.FUSION_ENCODER_L2_NORMALIZED_CONCATENATION
    assert sweep.view1_refinement_policy == frozen.view1_refinement_policy
    assert sweep.min_training_steps == frozen.min_training_steps
    mvdec.validate_imported_view1_run_gate(sweep, "REUTERS", 57, 1)
    with pytest.raises(ValueError):
        mvdec.validate_imported_view1_run_gate(sweep, "RCV1", 42, 1)
    with pytest.raises(ValueError):
        mvdec.validate_imported_view1_run_gate(frozen, "REUTERS", 57, 1)


def test_frozen_view1_loss_and_schedule_arms_change_one_setting(monkeypatch):
    _, mvdec = _load_mvdec(monkeypatch)
    frozen = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_PROTOCOL
    arms = {
        "L4_ONLY": {
            "reconstruction_weight": 0.0,
            "kmeans_weight": 0.0,
            "scatter_trace_weight": 0.0,
        },
        "LOSS_TENTH": {"kmeans_weight": 0.1, "scatter_trace_weight": 0.1},
        "REFRESH_50": {
            "kmeans_refresh_interval": 50,
            "kmeans_refresh_policy": mvdec.ALT_KMEANS_REFRESH_POLICY,
        },
        "REFINE_LR_1E4": {"refinement_learning_rate": 1e-4},
    }
    for key, changed in arms.items():
        arm = getattr(mvdec, f"MIN_REFINEMENT_FROZEN_VIEW1_{key}_PROTOCOL")
        assert arm.claim_scope.startswith("Controlled diagnostic")
        for field, value in changed.items():
            assert getattr(arm, field) == value
        for field in (
            "view1_refinement_policy",
            "min_training_steps",
            "fusion_contract",
            "input_normalization",
        ):
            assert getattr(arm, field) == getattr(frozen, field)
        assert mvdec.PUBLIC_PROTOCOLS[arm.protocol_id] is arm
        assert (
            mvdec.protocol_method_name(arm)
            == data.PUBLIC_PROTOCOL_METHODS[arm.protocol_id]
        )
        mvdec.validate_imported_view1_run_gate(arm, "REUTERS", 57, 1)
    assert frozen.refinement_learning_rate == 0.0
    assert "optimizer" not in frozen.manifest_contract(0.001)
    lr = mvdec.MIN_REFINEMENT_FROZEN_VIEW1_REFINE_LR_1E4_PROTOCOL
    assert lr.manifest_contract(0.001)["optimizer"] == {
        "refinement_learning_rate": 1e-4
    }
    with pytest.raises(ValueError, match="refinement_learning_rate"):
        mvdec.replace(frozen, refinement_learning_rate=-1.0)
