# Contributing

Thank you for your interest in contributing!

## How to Contribute

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/my-feature`)
3. **Commit** your changes (`git commit -m 'Add new feature'`)
4. **Push** to the branch (`git push origin feature/my-feature`)
5. **Open** a Pull Request

## Guidelines

- Follow [PEP 8](https://pep8.org/) style conventions
- Format code with [Black](https://github.com/psf/black)
- Add tests for new functionality
- Update documentation as needed
- Keep commits focused and atomic

## Development Setup

```bash
git clone https://github.com/regerah/Taillor-Protecting-Principal-Components-Parameterefficient.git
cd Taillor-Protecting-Principal-Components-Parameterefficient
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running Tests

```bash
pytest --cov=. --cov-report=term-missing
```

## Code of Conduct

Be respectful, constructive, and collaborative.
