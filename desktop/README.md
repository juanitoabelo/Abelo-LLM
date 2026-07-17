# Desktop App (Tauri v2)

## Prerequisites
- Rust toolchain: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Tauri system deps: `brew install gdk-pixbuf gtk+3 libsoup`

## Development
```bash
cd desktop/src-tauri
cargo tauri dev
```

## Build
```bash
cargo tauri build
```

The built app will be in `desktop/src-tauri/target/release/bundle/`.
