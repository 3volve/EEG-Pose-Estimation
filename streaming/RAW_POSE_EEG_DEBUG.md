# Raw pose/EEG diagnostic sidecar

The experimental use of these sidecars is governed by the project-level
[`DATA_PIPELINE_RESEARCH_PLAN.md`](../DATA_PIPELINE_RESEARCH_PLAN.md).

Pass `--save-raw-pose-eeg-debug` to `collect-paired`, `profile-build`, or
`bootstrap-base` to write an additional diagnostic archive next to the normal
training archive. The normal archive and model-training contract are unchanged.

For example:

```powershell
python __main__.py bootstrap-base `
  --user-id Evo `
  --out eeg_base_corrected_v1.pt `
  --save-raw-pose-eeg-debug
```

If the training archive is `paired_profile_session.npz`, the sidecar is named
`paired_profile_session.raw_pose_eeg_debug.npz`.

## Contents

The `raw-pose-eeg-debug-v1` archive contains:

- `eeg_native_*`: every unfiltered LSL sample column, reconstructed monotonic
  sample time, original LSL source timestamp, and chunk boundaries.
- `eeg_packet_*`: the selected and filtered overlapping EEG packets actually
  presented to the model, with packet IDs and start/end times.
- `pose_native_*`: every MediaPipe callback result at native pose rate, including
  all 33 image and world landmarks as `(x, y, z, visibility, presence)`, image
  dimensions, detection status, capture timestamp, and result-received time.
- `pose_processed_*`: every normalized/smoothed 48-feature pose vector processed
  by the live training path, its 24-dimensional latent, decoder reconstruction,
  confidence, reconstruction error, and exact native-pose frame index.
- `paired_*`: exact EEG-packet indices, the two bracketing processed-pose frame
  indices, interpolation weight, interpolated raw feature vector, latent,
  reconstruction, and quality scores used for each training target.
- `guide_block_*`: bootstrap/profile guide boundaries, movement names, data-split
  roles, acceptance results, and the complete block diagnostics JSON.
- `metadata_json`: EEG preprocessing, pose model/checkpoint, camera and mirroring
  configuration, smoothing, pose feature source, pairing tolerance, and the
  corresponding normal training archive.

The sidecar intentionally does not store camera images or video. All MediaPipe
outputs required to recompute normalized features, velocities, smoothing, pose
latents, and alternate EEG/pose pairings are retained without the privacy and
storage cost of raw video.

For bootstrap captures, `guide_block_role == "test"` remains the permanent
holdout. Do not use those blocks to select a representation, latent size,
smoothing rule, or training configuration; develop with support/query/validation
blocks and open the test role only for the final locked comparison.

## Live pairing clock (September 2026)

New captures set `pairing_pose_time_basis: capture_timestamp_ms` in metadata.
Live interpolation, its gap limit, and sidecar pairing reconstruction use
`timestamp_ms / 1000`, in Python monotonic seconds. This timestamp is assigned
immediately after a successful camera read, before conversion and inference;
it is a camera-delivery timestamp, not a hardware exposure timestamp.
Callback receipt times remain saved for latency diagnostics. Older archives
without the marker used callback receipt time; pairing verification respects
that historical convention. Existing recordings are not rewritten.

Live EEG timestamps now use LSL clock synchronization, dejittering and
monotonicity processing, then a measured offset to Python monotonic time.
Camera buffering and causal pose smoothing can still add physical signal delay.

## Corrected EEG timing and independent raw capture

Live reads request at most five samples with a 20 ms timeout. Model windows
remain 200 samples with a 50-sample stride. The receiver enables
`proc_clocksync | proc_dejitter | proc_monotonize`; it does not add LSL's
`time_correction()` a second time. Invalid or nonadvancing corrected timestamps
surface as an ingress error rather than being silently rewritten.

`eeg_timing_version` in recording metadata identifies this convention separately
from the unchanged EEG channel/filter preprocessing signature. Existing model
weights remain loadable, but old and new training labels have different timing.

The same-host LSL-to-Python clock mapping uses the shortest of ten bracketed
clock reads at startup. That offset stays fixed for the session so clock-reading
noise cannot create new timing steps. Every five seconds, diagnostic clock
measurements record the newly measured offset, read uncertainty, applied offset,
and source clock correction. These measurements allow drift to be checked;
this implementation does not dynamically correct drift between the two local
clocks. LSL itself tracks source-to-local clock correction.

With diagnostic capture enabled:

- `eeg_native_samples` is the unfiltered data used by the processed inlet;
  `eeg_native_monotonic_time_s` contains the mapped times used by model packets.
- `eeg_native_corrected_lsl_time_s` contains synchronized/dejittered LSL times.
- `eeg_native_source_time_s` is NaN for the processed inlet: the original
  timestamps cannot be recovered from this inlet after LSL postprocessing.
- `eeg_raw_lsl_samples`, `eeg_raw_lsl_source_time_s`, and the corresponding chunk
  start/length/receipt arrays contain an independent unprocessed inlet capture.
  It may include different startup/end samples and chunk boundaries. Match
  sample values before comparing; never zip these arrays by row or chunk number.
- `eeg_clock_measurements_json` records clock-map diagnostics.

Legacy archives retain their original fields and interpretation. New source-time
analyses must use the independent raw inlet, not the NaN placeholder.
No raw inlet is opened when raw diagnostics are disabled.

LSL smoothing needs settling time (documentation estimates 30-120 seconds in
worst cases for sub-millisecond jitter). No fixed warmup exclusion or gap-window
exclusion is imposed by this change. A short recording tests transport and basic
timing behavior but cannot demonstrate fully settled synchronization accuracy.

### Gap policy remains unchanged

A single missing sample affects several overlapping windows. Removing those
windows would reduce training coverage, and the live predictor currently stacks
successive packets without checking continuity. Simply dropping packets could
therefore splice disconnected periods into one context. Resetting context and
filter state has its own warmup/transient consequences. These decisions require
a coordinated policy and trustworthy loss indicators; the current LSL EEG-only
stream does not expose the hardware sample counter. This implementation retains
samples/windows and raw evidence and makes no claim to repair missing samples.
A later policy can distinguish an isolated missing sample from a longer outage,
report affected windows, and evaluate interpolation or exclusion on development
data before changing the training contract.

References:
https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/enums.html
https://labstreaminglayer.readthedocs.io/projects/liblsl/ref/inlet.html
https://labstreaminglayer.readthedocs.io/info/time_synchronization.html

The tested `synaptech-arm` Windows Python uses GetTickCount64 for monotonic
(resolution 15.625 ms), while LSL uses a finer clock. Clock-read uncertainty
includes half the reported Python clock resolution rather than claiming zero
uncertainty for reads within the same tick. This limits the existing pose clock;
a future shared high-resolution clock migration must change every acquisition
and pairing timestamp consistently, not just EEG. No such migration is included.
