#!/usr/bin/env python3
"""Generate a data-flow diagram for the latent-adapter injection solution.

Mirrors the style of the existing X-ray guidance pipeline figure: light-gray
boxes for components, white plaintext boxes for function calls / file:line
annotations, dotted edges for the reward/target feed.

Color legend
------------
  green   : NEW code (models/latent_adapter.py)            -- Step 1
  orange  : WIRING (one-line change in an existing file)
  gray    : UNCHANGED existing pipeline (zero edits)
  yellow  : the injection seam (inside LatentAdaptedWrapper.featurize)
  blue/dashed : Step 2 (future) -- swap the injector, add training

Usage
-----
    pip install graphviz          # python binding
    # also needs the Graphviz system package providing `dot`:
    #   macOS:  brew install graphviz
    #   debian: apt-get install graphviz
    python generate_solution_diagram.py            # -> latent_adapter_solution.png + .gv

If the `graphviz` python package or `dot` is unavailable, the script still
writes the .gv source; render it yourself with:
    dot -Tpng latent_adapter_solution.gv -o latent_adapter_solution.png
"""

from __future__ import annotations

import sys

OUT = "latent_adapter_solution"

# --- styles ------------------------------------------------------------------
NEW = dict(style="filled", fillcolor="#d5f5e3", color="#1e8449", shape="box")
WIRING = dict(style="filled", fillcolor="#fdebd0", color="#b9770e", shape="box")
UNCHANGED = dict(style="filled", fillcolor="#eeeeee", color="#888888", shape="box")
SEAM = dict(style="filled", fillcolor="#fcf3cf", color="#b7950b", shape="box")
FUTURE = dict(style="dashed,filled", fillcolor="#d6eaf8", color="#2471a3", shape="box")
NOTE = dict(shape="plaintext", fontsize="9")  # white annotation boxes
CLUSTER_ATTRS = dict(style="rounded", color="#bbbbbb", fontsize="11", labeljust="l")


def build(include_step2: bool = True, out: str = OUT, formats=("png", "svg")):
    try:
        from graphviz import Digraph
    except ImportError:
        _emit_raw_dot()
        return

    g = Digraph(out)
    # splines="spline" (default) so edge labels render; "ortho" drops them.
    g.attr(rankdir="TB", fontname="Helvetica", splines="spline", nodesep="0.45", ranksep="0.6")
    g.attr("node", fontname="Helvetica", fontsize="10", margin="0.12,0.08")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#555555")

    # ---------------------------------------------------------------- inputs
    with g.subgraph(name="cluster_inputs") as c:
        c.attr(label="INPUTS  (reused, unchanged)", **CLUSTER_ATTRS)
        c.node("struct_in", "args.structure (.pdb / .cif)\natomworks.parse(...)", **UNCHANGED)
        c.node("struct", "structure: dict\n(asym_unit, chain_info, _<model>_config)", **UNCHANGED)
        c.node("density_in", "args.density + resolution\nXMap.fromfile(...)", **UNCHANGED)
        c.node(
            "reward",
            "reward_function = RealSpaceRewardFunction\nholds target density + loss (L1/MSE)\n"
            "rewards/real_space_density.py",
            **UNCHANGED,
        )
        c.node("struct_in_note", "get_reward_function_and_structure()\nguidance_script_utils.py:225", **NOTE)
        c.edge("struct_in", "struct")
        c.edge("density_in", "reward")
        c.edge("struct_in_note", "reward", style="invis")

    # ----------------------------------------------------------- wiring layer
    with g.subgraph(name="cluster_wiring") as c:
        c.attr(label="WIRING  (one-line change)", **CLUSTER_ATTRS)
        c.node(
            "wire",
            "get_model_and_device()\nguidance_script_utils.py:165\n\n"
            "model = LatentAdaptedWrapper(\n"
            "    BoltzWrapper(...),   # or Protenix / RF3\n"
            '    AttrLatentIO("s"),   # "s_trunk" for Protenix/RF3\n'
            "    AffineInjector())    # Step 1",
            **WIRING,
        )

    # -------------------------------------------------- NEW adapter (Step 1)
    with g.subgraph(name="cluster_adapter") as c:
        c.attr(label="NEW  models/latent_adapter.py  (Step 1)", **CLUSTER_ATTRS)
        c.node(
            "wrapper",
            "LatentAdaptedWrapper[C]\n(decorator; satisfies FlowModelWrapper)\n"
            "fields: inner, latent_io, injector, training_adapter",
            **NEW,
        )
        c.node(
            "io",
            'AttrLatentIO(single_attr)\nread_single / write_single\n'
            "the ONLY model-specific knowledge\n"
            '(one string: "s" | "s_trunk")',
            **NEW,
        )
        c.node("injector", "AffineInjector  (pseudo-MLP)\ns′ = k·s + b\nk=1,b=0 → identity", **NEW)
        c.node("protocols", "Protocols\nLatentIO  /  LatentInjector\n(Step-2 swap points)", **NEW)
        c.edge("wrapper", "io", style="dashed", arrowhead="none", label="uses")
        c.edge("wrapper", "injector", style="dashed", arrowhead="none", label="uses")
        c.edge("protocols", "injector", style="invis")

    # --------------------------------------------- the injection seam (flow)
    with g.subgraph(name="cluster_seam") as c:
        c.attr(label="THE SEAM  LatentAdaptedWrapper.featurize()", **CLUSTER_ATTRS)
        c.node("f1", "1. feats = inner.featurize(structure)\n→ GenerativeModelInput[C]\n(conditioning: s, z, …)", **SEAM)
        c.node("f2", "2. s = latent_io.read_single(cond)", **SEAM)
        c.node("f3", "3. s′ = injector(s)   # k·s + b", **SEAM)
        c.node(
            "f4",
            "4. if not training_adapter:\n       s′ = s′.detach()\n"
            "# gradient isolation: guidance hits coords only",
            **SEAM,
        )
        c.node("f5", "5. cond′ = latent_io.write_single(cond, s′)\n# dataclasses.replace → sidecar state preserved", **SEAM)
        c.node("f6", "6. return GenerativeModelInput(x_init, cond′)", **SEAM)
        for a, b in [("f1", "f2"), ("f2", "f3"), ("f3", "f4"), ("f4", "f5"), ("f5", "f6")]:
            c.edge(a, b)

    # ------------------------------------------- downstream (UNCHANGED)
    with g.subgraph(name="cluster_down") as c:
        c.attr(label="DOWNSTREAM  (UNCHANGED — zero edits)", **CLUSTER_ATTRS)
        c.node(
            "pg",
            "PureGuidance.sample()\npure_guidance.py:74\nfeatures = model.featurize(structure)  # once\n"
            "for i in range(num_steps): sampler.step(… features=features)",
            **UNCHANGED,
        )
        c.node(
            "edm",
            "AF3EDMSampler.step()\nedm.py:301-487\nnoisy_state.requires_grad_(True)  # coords only",
            **UNCHANGED,
        )
        c.node(
            "mstep",
            "model.step(x_t, t, features=features)\nboltz/wrapper.py:874\ncond = features.conditioning  ← reads cond′\n"
            "(Protenix detaches cond when grad_needed)",
            **UNCHANGED,
        )
        c.node("guid", "reward.backward() on COORDS\n→ guidance (B,N,3) → guided Euler delta", **UNCHANGED)
        c.edge("pg", "edm")
        c.edge("edm", "mstep")
        c.edge("mstep", "guid")
        c.edge("guid", "edm", label="next step", constraint="false", style="dotted")

    # ------------------------------------------------ Step 2 (future)
    if include_step2:
        with g.subgraph(name="cluster_step2") as c:
            c.attr(label="STEP 2  (future — same seam, swap pieces)", **CLUSTER_ATTRS)
            c.node("mlp", "MLPInjector / FiLMInjector\nzero-init last layer → identity at start", **FUTURE)
            c.node("dens", "DensityInjector\nforward(s, e)   e = reward features\n(AlphaSAXS-style conditioning)", **FUTURE)
            c.node(
                "latentguid",
                "LatentGuidance scaler\ntraining_adapter=True\none-step-denoiser proxy\noptimize injector.parameters()",
                **FUTURE,
            )
            c.edge("mlp", "dens", style="invis")
            c.edge("dens", "latentguid", style="invis")

    # ------------------------------------------------ verification ladder
    g.node(
        "ladder",
        "VERIFICATION LADDER\n"
        "1 transform correct .............. ✅ test (k≠1)\n"
        "2 zero-impact at identity ........ ✅ test (k=1,b=0)\n"
        "3 gradient isolation ............. ✅ test (detach/attach)\n"
        "4 real model consumes cond′ ....... ⛳ cluster smoke run\n"
        "5 injection improves data fit .... \U0001f51c Step 2",
        shape="note",
        style="filled",
        fillcolor="#fef9e7",
        fontsize="9",
    )

    # ---------------------------------------------------------- cross edges
    g.edge("struct", "wire")
    g.edge("wire", "wrapper", label="constructs")
    g.edge("wrapper", "f1", label="featurize()")
    g.edge("io", "f2", style="dotted", arrowhead="none")
    g.edge("injector", "f3", style="dotted", arrowhead="none")
    g.edge("f6", "pg", label="GenerativeModelInput[C]\n(conditioning = cond′)")
    g.edge("reward", "guid", style="dotted", label="target density")
    # Step 2 swaps
    if include_step2:
        g.edge("injector", "mlp", style="dashed", color="#2471a3", label="swap")
        g.edge("reward", "dens", style="dashed", color="#2471a3", label="feeds e")
        g.edge("latentguid", "wrapper", style="dashed", color="#2471a3", label="flips training_adapter")
    g.edge("guid", "ladder", style="invis")

    written = []
    for fmt in formats:
        g.format = fmt
        written.append(g.render(out, cleanup=True))
    print("Wrote " + ", ".join(written))


def build_downstream_impact(out: str = OUT + "_downstream", formats=("png", "svg")):
    """Upstream is a BLACK BOX; expand what a swapped s' does downstream.

    Every downstream consumer is tagged by impact category:
      green  INTENDED   -- the effect we want
      gray   INVARIANT  -- provably unaffected (derives from atom arrays / x_init, not s)
      orange INDIRECT   -- same mechanism, but now operating on a shifted trajectory
      red    RISK        -- can break; needs a guard
    """
    try:
        from graphviz import Digraph
    except ImportError:
        _emit_raw_dot()
        return

    INTENDED = dict(style="filled", fillcolor="#d5f5e3", color="#1e8449", shape="box")
    INVARIANT = dict(style="filled", fillcolor="#eeeeee", color="#888888", shape="box")
    INDIRECT = dict(style="filled", fillcolor="#fdebd0", color="#b9770e", shape="box")
    RISK = dict(style="filled,bold", fillcolor="#f5b7b1", color="#c0392b", shape="box")

    g = Digraph(out)
    g.attr(rankdir="TB", fontname="Helvetica", splines="spline", nodesep="0.4", ranksep="0.55")
    g.attr("node", fontname="Helvetica", fontsize="10", margin="0.12,0.08")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#555555")

    # --- black box upstream + the single injection point --------------------
    g.node(
        "blackbox",
        "UPSTREAM  (BLACK BOX — assumed correct)\n"
        "inner.featurize(structure) → encoder / trunk\n"
        "→ conditioning with single representation  s   [B, tokens, d_s]",
        shape="box3d",
        style="filled",
        fillcolor="#2c3e50",
        fontcolor="white",
    )
    g.node(
        "inject",
        "INJECTION  (the only change)\n"
        "s′ = k·s + b   (identical shape / dtype / device)\n"
        "detach() in sampling mode",
        style="filled",
        fillcolor="#d5f5e3",
        color="#1e8449",
        shape="box",
        penwidth="2",
    )
    g.edge("blackbox", "inject", label="s")

    # --- INTENDED ------------------------------------------------------------
    with g.subgraph(name="cluster_intended") as c:
        c.attr(label="INTENDED EFFECT", **CLUSTER_ATTRS)
        c.node("step", "model.step(): diffusion forward\nuses s_trunk=cond.s′  (boltz/wrapper.py:874)\n→ x̂₀ changes", **INTENDED)
        c.node("reuse", "cached once, reused across all 200 steps\n→ a CONSTANT bias that compounds over the trajectory", **INTENDED)
        c.node("xout", "sampled coordinates / ensemble shift\n(the steering we want)", **INTENDED)
        c.edge("step", "xout")
        c.edge("reuse", "step", style="dotted")

    # --- INVARIANT (safe) ----------------------------------------------------
    with g.subgraph(name="cluster_inv") as c:
        c.attr(label="PRESERVED INVARIANTS  (provably unaffected by s′)", **CLUSTER_ATTRS)
        c.node("shapes", "tensor shapes / atom counts / x_init\n(s is [B,tokens,d], not coords → no shape change)", **INVARIANT)
        c.node("recon", "process_structure_to_trajectory_input\nreconciler · model_atom_array · RewardInputs\n(derive from atom arrays, NOT s)", **INVARIANT)
        c.node("bf", "b_factors / occupancies / elements\n(from atom arrays)", **INVARIANT)
        c.node("det", "determinism / seeding\n(injection adds no randomness)", **INVARIANT)
        c.node("out_shape", "GuidanceOutput / trajectory / save_everything\nSAME shapes, different values", **INVARIANT)

    # --- INDIRECT ------------------------------------------------------------
    with g.subgraph(name="cluster_indirect") as c:
        c.attr(label="INDIRECT EFFECTS  (mechanism unchanged, inputs shifted)", **CLUSTER_ATTRS)
        c.node("reward", "reward on COORDS, grad on coords only\n(cond detached) → structurally identical,\nbut evaluated on shifted x̂₀", **INDIRECT)
        c.node("guid", "guidance / guided Euler delta\n(edm.py) → different direction", **INDIRECT)
        c.node("fk", "FK steering: resampling weights\n→ particle distribution reweighted", **INDIRECT)
        c.edge("reward", "guid")

    # --- RISK ----------------------------------------------------------------
    with g.subgraph(name="cluster_risk") as c:
        c.attr(label="RISKS / FAILURE MODES  (add a guard)", **CLUSTER_ATTRS)
        c.node("nan", "OOD s′ (large k or b) → unstable x̂₀ / NaN\npropagates to reward → whole trajectory\nGUARD: clamp |Δ|, assert_no_nans on step output", **RISK)
        c.node("leak", "training_adapter=True during sampling\n→ autograd leak (Boltz step does NOT detach)\nGUARD: detach-in-sampling (already in wrapper)", **RISK)
        c.node("collapse", "uniform bias across ensemble dim\n→ diversity collapse (loses flexibility)\nGUARD: per-member / noise-conditioned injection", **RISK)

    # --- propagation edges from injection ------------------------------------
    g.edge("inject", "step", label="s′ via features.conditioning", penwidth="2")
    g.edge("inject", "reuse", style="dotted")
    g.edge("inject", "shapes", style="dashed", arrowhead="empty", label="no effect")
    g.edge("inject", "recon", style="dashed", arrowhead="empty", label="no effect")
    g.edge("xout", "reward", label="shifted coords")
    g.edge("guid", "step", label="next step", constraint="false", style="dotted")
    g.edge("inject", "nan", color="#c0392b")
    g.edge("inject", "leak", color="#c0392b", style="dashed")
    g.edge("step", "collapse", color="#c0392b", style="dotted")
    g.edge("xout", "out_shape", style="dashed", arrowhead="empty")

    # --- legend --------------------------------------------------------------
    g.node(
        "legend",
        "IMPACT LEGEND\n"
        "green  INTENDED — desired steering\n"
        "gray   INVARIANT — provably safe (no s dependence)\n"
        "orange INDIRECT — shifted inputs, same mechanism\n"
        "red    RISK — needs a guard",
        shape="note",
        style="filled",
        fillcolor="#fef9e7",
        fontsize="9",
    )

    written = []
    for fmt in formats:
        g.format = fmt
        written.append(g.render(out, cleanup=True))
    print("Wrote " + ", ".join(written))


def build_final_recommendation(out: str = OUT + "_final", formats=("png", "svg")):
    """Finalized design: a DECOUPLED latent-optimization pre-pass + FROZEN sampler.

    See latent_space_optimization.md §4. Targets the post-trunk single rep
    (s_trunk); the diffusion sampler is unchanged. ``m`` is never touched.
    """
    try:
        from graphviz import Digraph
    except ImportError:
        _emit_raw_dot()
        return

    NEW = dict(style="filled", fillcolor="#d5f5e3", color="#1e8449", shape="box")
    FROZEN = dict(style="filled", fillcolor="#eeeeee", color="#888888", shape="box")
    CHOICE = dict(style="filled", fillcolor="#fcf3cf", color="#b7950b", shape="box")

    g = Digraph(out)
    g.attr(rankdir="TB", fontname="Helvetica", splines="spline", nodesep="0.45", ranksep="0.6")
    g.attr("node", fontname="Helvetica", fontsize="10", margin="0.12,0.08")
    g.attr("edge", fontname="Helvetica", fontsize="9", color="#555555")

    with g.subgraph(name="cluster_pre") as c:
        c.attr(label="DECOUPLED LATENT-OPT PRE-PASS  (NEW — runs once)", **CLUSTER_ATTRS)
        c.node("F", "inner.featurize(structure)\ntrunk runs ONCE → caches s_trunk, z_trunk", **NEW)
        c.node("D", "DeltaInjector\ndelta = nn.Parameter(zeros)  → identity at init\ns' = s_trunk + delta", **NEW)
        c.node("OBJ", "objective(s')  (no sampler unroll)", **CHOICE)
        c.node("A", "(a) aux head h(s') → observable\nvs experimental target\nsampler NOT called", **NEW)
        c.node("B", "(b) ONE model.step(x_t,t,feats) → x̂₀\nreuse real_space_density reward\nsingle denoiser call, not the loop", **NEW)
        c.node("UPD", "loss.backward() → grad on delta ONLY\nopt.step()  (repeat opt_steps)\nregularize ‖delta‖", **NEW)
        c.node("WR", "feats' = write_single(feats,\n   (s_trunk+delta).detach())", **NEW)
        c.edge("F", "D"); c.edge("D", "OBJ")
        c.edge("OBJ", "A"); c.edge("OBJ", "B")
        c.edge("A", "UPD"); c.edge("B", "UPD")
        c.edge("UPD", "WR", label="converged")

    with g.subgraph(name="cluster_samp") as c:
        c.attr(label="FROZEN SAMPLER  (UNCHANGED — zero edits)", **CLUSTER_ATTRS)
        c.node("PG", "PureGuidance.sample()\nfeatures = feats' (fixed across steps)", **FROZEN)
        c.node("EDM", "AF3EDMSampler.step()\nnoisy_state.requires_grad_(True)  # COORDS only", **FROZEN)
        c.node("MS", "model.step(x_t, t, features=feats')\nreads optimized s_trunk", **FROZEN)
        c.node("G", "reward.backward() on coords\n→ guided Euler delta", **FROZEN)
        c.edge("PG", "EDM"); c.edge("EDM", "MS"); c.edge("MS", "G")
        c.edge("G", "EDM", label="next step", constraint="false", style="dotted")

    g.edge("WR", "PG", label="optimized features", penwidth="2")

    g.node(
        "note",
        "EASE (post-trunk latent): RF3 / Boltz1 = clean;\n"
        "Protenix = bypass step() detach + recompute pair_z;\n"
        "Boltz2 = recompute diffusion_conditioning.\n"
        "m is NEVER touched — its info is already in s_trunk / z_trunk.",
        shape="note", style="filled", fillcolor="#fef9e7", fontsize="9",
    )
    g.edge("WR", "note", style="invis")

    written = []
    for fmt in formats:
        g.format = fmt
        written.append(g.render(out, cleanup=True))
    print("Wrote " + ", ".join(written))


def _emit_raw_dot():
    """Fallback: graphviz python package missing -> write a .gv the user can render."""
    print("graphviz python package not found; writing raw DOT instead.", file=sys.stderr)
    print(f"Render with:  dot -Tpng {OUT}.gv -o {OUT}.png", file=sys.stderr)
    # Minimal raw-DOT mirror of the structure above.
    dot = '''digraph latent_adapter_solution {
  rankdir=TB; node [shape=box, style=filled, fontname=Helvetica];
  struct  [label="structure: dict", fillcolor="#eeeeee"];
  reward  [label="RealSpaceRewardFunction\\n(target density)", fillcolor="#eeeeee"];
  wire    [label="get_model_and_device()\\nmodel = LatentAdaptedWrapper(\\n  BoltzWrapper(...), AttrLatentIO(\\"s\\"), AffineInjector())", fillcolor="#fdebd0"];
  wrapper [label="LatentAdaptedWrapper[C]\\n(satisfies FlowModelWrapper)", fillcolor="#d5f5e3"];
  io      [label="AttrLatentIO(\\"s\\"|\\"s_trunk\\")\\nonly model-specific knowledge", fillcolor="#d5f5e3"];
  injector[label="AffineInjector  s'=k*s+b\\nk=1,b=0 -> identity", fillcolor="#d5f5e3"];
  seam    [label="featurize: read s -> inject -> detach(if sampling) -> write s'", fillcolor="#fcf3cf"];
  pg      [label="PureGuidance.sample() / AF3EDMSampler.step()\\nUNCHANGED", fillcolor="#eeeeee"];
  mstep   [label="model.step(): cond = features.conditioning (reads s')", fillcolor="#eeeeee"];
  step2   [label="STEP 2: MLP/FiLM/Density injector + LatentGuidance training", fillcolor="#d6eaf8", style="dashed,filled"];
  struct -> wire -> wrapper -> seam -> pg -> mstep;
  wrapper -> io [style=dashed,arrowhead=none]; wrapper -> injector [style=dashed,arrowhead=none];
  reward -> mstep [style=dotted,label="target density"];
  injector -> step2 [style=dashed,label="swap"];
}
'''
    with open(f"{OUT}.gv", "w") as fh:
        fh.write(dot)
    print(f"Wrote {OUT}.gv")


if __name__ == "__main__":
    # Full map (PNG + SVG)
    build(include_step2=True, out=OUT, formats=("png", "svg"))
    # Compact current-state view: Step 1 only (PNG + SVG)
    build(include_step2=False, out=OUT + "_step1", formats=("png", "svg"))
    # Downstream-impact view: upstream as black box, downstream expanded
    build_downstream_impact(out=OUT + "_downstream", formats=("png", "svg"))
    # Finalized decoupled latent-optimization pre-pass + frozen sampler
    build_final_recommendation(out=OUT + "_final", formats=("png", "svg"))
