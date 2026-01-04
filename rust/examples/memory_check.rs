use your_ai_rs::utils::MemoryInfo;

fn main() {
    match MemoryInfo::current() {
        Ok(info) => {
            println!("Memory Info Check:");
            println!("RSS: {}", info.rss_formatted());
            println!("Total: {}", info.total_formatted());
            println!("Available: {}", info.available_formatted());
            println!("Usage: {:.2}%", info.usage_percentage());

            if info.system_total_bytes == 0 {
                println!("ERROR: Total bytes is 0!");
                std::process::exit(1);
            }
            if info.system_available_bytes == 0 {
                println!("ERROR: Available bytes is 0!");
                std::process::exit(1);
            }
        }
        Err(e) => {
            println!("Failed to get memory info: {}", e);
            std::process::exit(1);
        }
    }
}
