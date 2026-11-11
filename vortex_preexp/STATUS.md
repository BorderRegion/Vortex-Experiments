# Completed pre-experiment

Decision: MODIFY. Authoritative report: reports/PREEXP_REPORT.md; raw final reasoning: results/raw/final_disposition.json.

Remote directory: /root/autodl-tmp/vortex_preexp on the user-authorized MTT S4000 host. Existing system /usr/bin/python, torch_musa and MUSA 5.2 used; no driver or SDK replacement. Unrelated /root/wcc was not touched.

Completed: P0 device/math/spectral checks and throughput pilot; P1 isotropic/outlier and trained-QK distributions, plus actual first-layer causal Q/K/V replay; P2 all six original runs, 61.44M tokens each; P2B all three associative continuations to 307.2M total tokens each; dispatch and CPU/MUSA sparse-loss/SDPA gradient checks; P2C four-layer RoPE calibration, 61.44M tokens; P4 all specified prefill/decode shapes with synchronized wall and MUSA Events.

P2C failed its frozen positive-control gate (six-condition mean7.2266%, min5.0781%). No extra four-layer Vortex/ring_plain runs were required. B remains inconclusive for method-quality comparison because the RoPE control also failed. P3 skipped for the recorded experimental reasons; no real-text NLL or native 4-bit speedup is claimed. P1 actual attention replay contains partial numerical positives; these are preserved and not dismissed as a fully negative mechanism result.

GPU occupied time: 2.0319157 hours, including active-stage Python overhead and superseded P1 measurements, below36h. Conditional stages are not run solely to consume the nominal25–30h allocation. All GPU experiments ended. Checkpoint and raw files retained. Completed report/audit generation uses CPU only.

Source, configs, reports, raw data and checkpoints are mirrored locally. Uploading local stale results over the remote directory must be avoided. Final report regeneration: /usr/bin/python -m experiments.finalize (reads final_disposition.json; no training). audit.py checks evidence coverage and writes a SHA256 manifest; scientific claims must still be compared with raw data. Compact review ZIP excludes weights/QK .pt; full checkpoint copies remain in the full directory.
