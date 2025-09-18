# Contributing to Familiar GM Assistant

Thank you for your interest in contributing to Familiar! We welcome contributions from the community.

## 🚀 Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/familiar.git`
3. Create a feature branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Commit with clear messages: `git commit -m "feat: add new feature"`
6. Push to your fork: `git push origin feature/your-feature-name`
7. Create a Pull Request

## 📋 Development Setup

```bash
# Install dependencies
npm install
pip install -r requirements.txt

# Run development server
npm run dev

# Run tests
npm test
python -m pytest tests/

# Run linting
npm run lint
```

## 🧪 Testing Requirements

- All new features must include tests
- Maintain >80% code coverage
- Run theater detection: `python comprehensive_analysis_engine.py --mode theater-detection`
- Ensure NASA POT10 compliance (functions <60 lines)

## 📝 Code Style

### JavaScript/TypeScript
- Use ESLint configuration
- Follow existing patterns in codebase
- Use async/await over callbacks
- Document complex functions with JSDoc

### Python
- Follow PEP 8
- Use type hints where applicable
- NASA Rule 4: Functions under 60 lines
- NASA Rule 5: Comprehensive defensive assertions

## 🏗️ Architecture Guidelines

### Queen-Princess-Drone Hierarchy
- **Queen**: Central orchestration only
- **Princess**: Domain-specific coordination (6 domains)
- **Drone**: Task execution agents (21 specialized agents)

### File Organization
- `/src` - Source code
- `/tests` - Test files
- `/docs` - Documentation
- `/scripts` - Utility scripts
- `/config` - Configuration files

## 🎯 Commit Convention

Use conventional commits format:
- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Build process/auxiliary tool changes

## 🐛 Reporting Issues

1. Check existing issues first
2. Use issue templates
3. Provide:
   - Clear description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information
   - Error messages/logs

## 💡 Feature Requests

1. Check existing feature requests
2. Describe the problem it solves
3. Provide use cases
4. Consider implementation approach

## 🔍 Code Review Process

All PRs require:
1. Passing CI/CD checks
2. Theater detection score >60
3. Code review from maintainer
4. Updated documentation
5. Test coverage maintained

## 📜 Licensing

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Community Guidelines

- Be respectful and inclusive
- Provide constructive feedback
- Help others when possible
- Follow the Code of Conduct

## 📞 Contact

- Issues: [GitHub Issues](https://github.com/yourusername/familiar/issues)
- Discussions: [GitHub Discussions](https://github.com/yourusername/familiar/discussions)

## 🏆 Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for helping make Familiar better!