# Ubuntu Core: Build Your First Image — Summary

Source: https://documentation.ubuntu.com/core/tutorials/build-your-first-image/index.html

## Overview
Build a custom Ubuntu Core image for Raspberry Pi by creating a model assertion, signing it, compiling the image, writing it to a microSD card, and booting the device.

## Prerequisites
- Basic Linux/terminal familiarity.
- Hardware requirements for host/target (per Canonical’s recommended specs).

## Steps (High Level)
1) **Create Ubuntu One account**
   - Set up Ubuntu One SSO for snapcraft.io.
   - Export Snapcraft credentials.
   - Retrieve developer account ID (e.g., `snapcraft whoami`).

2) **Create model assertion**
   - Download reference model (example: `ubuntu-core-24-pi-arm64.json`).
   - Edit `my-model.json`:
     - Replace `authority-id` and `brand-id` with your developer/brand IDs.
     - Ensure required snaps are included (e.g., pi, pi-kernel, core24, snapd).
     - Add/update timestamp.

3) **Sign the model**
   - Create a GPG key via snapcraft.
   - Register the key with Ubuntu One.
   - Sign the model using `snap sign` to produce a model assertion.

4) **Build and write the image**
   - Install `ubuntu-image`.
   - Build the image using the signed model assertion (with `--allow-snapd-kernel-mismatch` when required).
   - Write the image to a microSD card (e.g., Raspberry Pi Imager).

5) **Boot the image**
   - Insert the microSD into the Raspberry Pi and boot.
   - Configure networking (Ethernet or Wi‑Fi).
   - Connect to the device as instructed (initial services may be exposed on port 3000 depending on snaps).

## Related subpages
- Requirements
- Access Ubuntu One
- Create a model
- Sign the model
- Build the image
- Boot the image

---
Last updated on the source page: Aug 07, 2025 (per page footer).
