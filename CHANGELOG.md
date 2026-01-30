# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.8 and 0.1.9] - 2025-01-30

### Added
- Comprehensive pytest test suite with unit and integration tests
- GitHub Actions CI/CD workflow for automated testing
- Support for Python 3.9, 3.10, 3.11, and 3.12
- `CONTRIBUTING.md` with development guidelines
- `CHANGELOG.md` following Keep a Changelog format
- Automated PyPI publishing via git tags

### Changed
- Updated documentation to use correct `keep=` parameter (not `mask=`)
- Fixed GitHub repository links in documentation

### Fixed
- Fixed docs/conf.py pointing to wrong repository
- Fixed test files using deprecated `mask=` parameter
- Fixed index alignment issues in EasyFlow.generate() method

## [0.1.7]

### Added
- `TablePValues` class for statistical significance testing
- Multiple testing correction options (none, bonferroni, fdr_bh)
- Improved input validation with helpful error messages
- Comprehensive docstrings for all public methods

### Changed
- Renamed `mask` parameter to `keep` in `add_exclusion()` for clarity
  - `keep=True` means the row is retained (kept)
  - `keep=False` means the row is excluded (removed)

### Deprecated
- `mask` parameter in `add_exclusion()` - use `keep` instead (will be removed in v0.2.0)

### Fixed
- Numerical stability in SMD calculations for edge cases
- Handling of variables with all missing values

## [0.1.1]

### Added
- Initial public release
- `EquiFlow` main class for cohort flow analysis
- `EasyFlow` simplified interface with method chaining
- `TableFlows` for cohort size tracking
- `TableCharacteristics` for demographic breakdowns
- `TableDrifts` for SMD calculations
- `FlowDiagram` for visual flow diagram generation
- Support for categorical, normal, and non-normal continuous variables
- Custom variable ordering and category limiting
- Flexible formatting options

## [0.1.0a2]

### Added
- Alpha release for testing

[Unreleased]: https://github.com/MoreiraP12/equiflow-v2/compare/v0.1.8...HEAD
[0.1.8]: https://github.com/MoreiraP12/equiflow-v2/compare/v0.1.7...v0.1.8
[0.1.7]: https://github.com/MoreiraP12/equiflow-v2/compare/v0.1.1...v0.1.7
[0.1.1]: https://github.com/MoreiraP12/equiflow-v2/compare/v0.1.0a2...v0.1.1
[0.1.0a2]: https://github.com/MoreiraP12/equiflow-v2/releases/tag/v0.1.0a2
