# TriArt

画像をポリゴンアート（三角形モザイク）風に変換するツール。

## 実行方法

```bash
python -m triart
```

## Development

CLI tools (`lefthook`) are managed by [aqua](https://aquaproj.github.io/) with versions pinned in [aqua.yaml](aqua.yaml).

### Install tools

Install aqua itself first (see the [aqua installation guide](https://aquaproj.github.io/docs/install)), then install the pinned tools:

```bash
aqua install
```

### Set up git hooks

[lefthook](lefthook.yml) runs ruff lint, ruff format, and pytest checks on staged `*.py` files before each commit. Register the hooks once after cloning:

```bash
lefthook install
```
