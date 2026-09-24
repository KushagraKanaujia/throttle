# Project State (2026-08-29)

## Recent Major Changes

### PyPI Publication (2026-08-29)
- **Package**: `throttle-pro` v0.3.0 published to PyPI
- **Install**: `pipx install throttle-pro` (recommended) or `pip install throttle-pro`
- **Repository**: https://github.com/KushagraKanaujia/throttle.git
- **Distribution**: Built and validated (191KB wheel, 349KB source tarball)

### Configuration File Support
- **Feature**: Optional `~/.throttle/config.yaml` for default values
- **Dependency**: Requires `pyyaml` (optional extra)
- **Implementation**: `src/throttle/config.py`
- **Behavior**: CLI flags always override config file values
- **Critical Fix**: Required arguments become optional when present in config (commit `4e2c777`)

### Installation Path
- **Primary**: pipx (avoids macOS Homebrew Python PEP 668 externally-managed-environment error)
- **Alternative**: pip inside virtualenv
- **Documentation**: README updated with pipx-first approach (commit `1622664`)

### Merged PRs
- **PR #21** (JohnScheuer): `throttle watch` command for real-time metrics monitoring
- **PR #22** (Waqas): validate-sim URL normalization fix + Kaggle dual-T4 validation notebook
- **PR #23** (Dhruv): Embeddings documentation in README

## Working Directory
- **Current**: `/tmp/throttle-clean` (clean clone from origin/main)
- **DO NOT USE**: `/private/tmp/throttle-analysis` (deleted - was corrupted, git object database broken)

## Verification Protocol

### Four-Step Check (João's Method)
When verifying PR claims or features:
1. **Grep**: Search code for claimed functionality
2. **ls**: Verify claimed files exist
3. **git log**: Check commit history for claimed changes
4. **Functional test**: Run actual commands to verify behavior

This protocol caught multiple false reports from the corrupted repository and is now standard for all verification work.

## Active Contributor Assignments

### Claimed Tasks
- **João**: Bitext harness implementation + config file verification
- **Waqas**: Heavy load testing on GPU infrastructure
- **Dhruv**: README embeddings documentation (completed in PR #23)
- **Tobi**: Adversarial compare testing
- **Shivam**: ONNX runtime investigation for embeddings performance

### Completed Tonight
- Config file support implementation
- Config bug fix (required=False for args in config)
- PyPI publication workflow
- PR #21, #22, #23 merged to main
- README updated with pipx install path
- **Adversarial audit and blocker fixes** (see below)

### Adversarial Audit Findings (2026-08-29)

Pre-company submission audit revealed one important UX issue and one false alarm:

**Fixed - Golden Protocol Config Conflict:**
- **Issue**: User's `~/.throttle/config.yaml` with `concurrency: [2, 4]` silently conflicted with golden protocol's single-value requirement
- **Root Cause**: Golden CLI defaulted silently instead of erroring explicitly
- **Fix**: Added validation at cli.py:2050-2063 that errors clearly when receiving multiple concurrency values
- **Impact**: Aligns with tool philosophy - make conflicts explicit rather than silently resolving them
- **False Alarm**: Initial audit incorrectly reported this as "21 failing tests" - tests pass in clean state

**Fixed - Watch Command Timeout:**
- **Issue**: `throttle watch` hung indefinitely when no vLLM server available
- **Fix**: Added connection error detection at cli.py:3657-3677 with graceful exit and helpful error message
- **Impact**: Better first-time user experience

**Test Suite Status:**
- 404 tests passing (100% of non-skipped tests)
- 1 test skipped (intentionally excluded)
- No known failures

## Recent Decisions

### Repository Management
- **Clean clone policy**: Always work from /tmp/throttle-clean going forward
- **Corrupted repo incident**: /private/tmp/throttle-analysis had broken git object database, causing false verification reports
- **Verification standard**: All claims must be verified against clean repository with four-step protocol

### Installation UX
- **Primary blocker identified**: macOS Homebrew Python blocks `pip install` with PEP 668 error
- **Solution**: Lead with pipx in all documentation
- **Impact**: Zero-friction install for macOS users (largest user segment)

### PyPI Strategy
- **Package name**: `throttle-pro` (avoid conflict with existing PyPI package `throttle`)
- **Entry point**: `throttle` command
- **Version**: 0.3.0
- **Distribution quality**: Both wheel and tarball passed `twine check`

## Next Session Priorities

### Not Started (Needs Owner)
- TBD - check open GitHub issues for unclaimed work

### Future Work (Deferred)
- Duration-bounded golden protocol positions (currently count-bounded only)
- Block-major counterbalanced scheduler for multi-load decision eligibility
- Additional backend validation (vLLM, SGLang, LMDeploy GPU verification)
- Production proxy features (auto-reload, metrics export)

## Key Files

### Configuration
- `src/throttle/config.py` - Config file loading
- `.throttle.yaml.example` - Example config file
- `~/.throttle/config.yaml` - User config location (optional)

### Distribution
- `pyproject.toml` - Package metadata
- `dist/throttle_pro-0.3.0-py3-none-any.whl` - Built wheel (local only)
- `dist/throttle_pro-0.3.0.tar.gz` - Source distribution (local only)

### Documentation
- `README.md` - Main docs (includes PyPI install, quickstart, config)
- `CONTRIBUTING.md` - Dev setup and contribution guide
- `docs/KNOWN_GAPS.md` - Explicit limitations and boundaries
- `docs/GOLDEN_PROTOCOL.md` - Six-position counterbalanced protocol spec

## Git Status
- **Branch**: main
- **Last commit**: `1622664` - Lead with pipx install
- **Uncommitted changes**: None
- **Unpushed commits**: None
- **Working tree**: Clean

All work is synchronized with origin/main at https://github.com/KushagraKanaujia/throttle.git
