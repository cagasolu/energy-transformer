# Contributing to SELYNE

Thanks for your interest in SELYNE.

## Contributor License Agreement

Before your contribution can be merged, you must sign the [Contributor License Agreement](CLA.md).

Signing is automated: when you open your first pull request, the CLA bot will post a link and ask you to confirm. It takes about 30 seconds.

**You keep ownership of your contribution.** The CLA does not assign your copyright or restrict you from using your own code elsewhere. It only grants the maintainer the rights needed to distribute your contribution as part of SELYNE.

### Why we require this

SELYNE is dual-licensed: the open-source distribution is GPLv3, and a separate commercial license is available for users who cannot comply with GPLv3. Without a CLA on file from every contributor, contributed code could only ever be distributed under GPLv3, which would break the dual-licensing model.

## How to contribute

1. Open an issue describing the change before writing code, so we can agree on the approach.
2. Fork the repository and create a branch.
3. Include tests for any behavioral change.
4. Open a pull request and sign the CLA when prompted.

## Scope notes

- Changes to the SELYNE interface — the module API and the certification entry points — are versioned and are not accepted casually. Open an issue first.
- New attention variants are welcome, but must supply an energy function and a sensitivity bound, and must reproduce the ablation protocol against the untied standard baseline.
- Reproducibility matters here: pin seeds, report the exact configuration, and state the hardware used.

## Questions

Open an issue, or for commercial licensing contact: **cangorkemsu@gmail.com**
