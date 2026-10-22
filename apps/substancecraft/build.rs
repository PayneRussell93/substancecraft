//! Embeds the app icon and version information in the Windows executable, so
//! Explorer, shortcuts, and the installer show the material cube.

fn main() {
    println!("cargo:rerun-if-changed=assets/substancecraft.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut resource = winresource::WindowsResource::new();
        resource
            .set_icon("assets/substancecraft.ico")
            .set("ProductName", "SubstanceCraft")
            .set("FileDescription", "SubstanceCraft");
        if let Err(error) = resource.compile() {
            println!("cargo:warning=could not embed the app icon: {error}");
        }
    }
}
