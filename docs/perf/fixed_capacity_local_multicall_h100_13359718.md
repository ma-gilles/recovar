# Shared fixed-capacity local multi-call gate — H100 job 13359718

## Decision

The default-off fixed-capacity host seam now covers every chronological local
call, including a partially filled tail call, while invoking the same mature
EM numeric wrapper. The two-call H100 gate is bitwise exact for prepared
operands, continuous scores/posteriors, discrete support/argmax state, repeats,
and assembled outer outputs in float32 and float64.

This is a correctness-only Phase-1 result. Each call still crosses the Python /
JAX boundary separately, so it makes no speed or default-promotion claim.

## Provenance

| Item | Value |
|---|---|
| Slurm job | `13359718` (`COMPLETED`, exit `0:0`) |
| Source | `231a0191cc17615fe97eef9a3894b93a824f5a64` |
| GPU / node | H100 `GPU-099c0d77-bb85-f2e9-f628-148b733c9176` / `della-h21g4` |
| Elapsed / MaxRSS | `00:01:24` / `1,681,696 KiB` |
| Focused tests | `7 / 7` pass; JUnit SHA-256 `55e52a8b0d8a28421a2ad5186419f7872a0d8192c0d998aa0b603dae808a005b` |
| Source manifest | `eb668b8bb28adb7ade5aefb9ec73f8bb8bfc85bc76eb5209035838286a528c3f` |
| CUDA library | `4c75200f37abc1f4fbd14bc9b88ae1a6c2889f778d5ae5b078c17e5162694ee3` |
| Harness result | SHA-256 `96374d0c934f870f6f20ef0edf38c3ab4dc924fcea5cf0879e3abce19c2fc840` |
| Artifact root | `/scratch/gpfs/GILLES/mg6942/fixed_capacity_local_score_gate_multicall_231a0191c_20260902T2145Z/` |

## Coverage

The fixture has two authoritative calls, three active images in physical order
`[1, 0, 2]`, image capacity two per call, radix 16, and one inactive call slot.
The second call contains one real image plus one padded image and therefore
exercises both active chronology and poison/inert-tail exclusion.

- 16 call-level comparisons pass exactly: default versus disabled seam,
  mature versus fixed, and repeated default/fixed execution, for both calls
  and both precision lanes.
- 12 assembled production-output comparisons pass exactly, including the
  independently uninstrumented production topology.
- Donated `Ft_y` and `Ft_ctf` objects remain fresh for every invocation.

## Next boundary

The next phase must reduce launches and synchronization, not merely replace
equal host arrays. It will carry the mature local numeric state through the
sealed call program behind one fixed-shape execution boundary. Admission still
requires exact support and decisions plus repeat-bounded continuous noise.
