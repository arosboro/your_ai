use std::io;

/// Memory usage information in bytes
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Resident Set Size (physical memory used)
    pub rss_bytes: u64,
    /// Virtual memory size
    pub virtual_bytes: u64,
    /// Total system memory
    pub system_total_bytes: u64,
    /// Available system memory
    pub system_available_bytes: u64,
}

impl MemoryInfo {
    /// Get current process memory usage
    pub fn current() -> io::Result<Self> {
        #[cfg(target_os = "macos")]
        {
            Self::from_macos()
        }

        #[cfg(target_os = "linux")]
        {
            Self::from_linux()
        }

        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            Err(io::Error::new(
                io::ErrorKind::Unsupported,
                "Memory monitoring not supported on this platform",
            ))
        }
    }

    #[cfg(target_os = "macos")]
    fn from_macos() -> io::Result<Self> {
        use std::process::Command;

        // Get process memory via ps
        let output = Command::new("ps")
            .args(&["-o", "rss,vsz", "-p", &std::process::id().to_string()])
            .output()?;

        let output_str = String::from_utf8_lossy(&output.stdout);
        let lines: Vec<&str> = output_str.lines().collect();

        if lines.len() < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Failed to parse ps output",
            ));
        }

        let values: Vec<&str> = lines[1].split_whitespace().collect();
        if values.len() < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Failed to parse memory values",
            ));
        }

        let rss_kb: u64 = values[0]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Failed to parse RSS"))?;
        let vsz_kb: u64 = values[1]
            .parse()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Failed to parse VSZ"))?;

        // Get system memory via sysctl
        let sys_output = Command::new("sysctl").args(&["hw.memsize"]).output()?;

        let sys_str = String::from_utf8_lossy(&sys_output.stdout);
        let total_bytes: u64 = sys_str
            .split(':')
            .nth(1)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(0);

        // Get memory pressure (approximation of available memory)
        let vm_output = Command::new("vm_stat").output()?;

        let vm_str = String::from_utf8_lossy(&vm_output.stdout);
        let mut free_pages = 0u64;
        let mut inactive_pages = 0u64;

        for line in vm_str.lines() {
            if line.starts_with("Pages free:") {
                free_pages = line
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.trim().trim_end_matches('.').parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("Pages inactive:") {
                inactive_pages = line
                    .split(':')
                    .nth(1)
                    .and_then(|s| s.trim().trim_end_matches('.').parse().ok())
                    .unwrap_or(0);
            }
        }

        // Page size is typically 4096 bytes on macOS
        let page_size = 4096u64;
        let available_bytes = (free_pages + inactive_pages) * page_size;

        Ok(Self {
            rss_bytes: rss_kb * 1024,
            virtual_bytes: vsz_kb * 1024,
            system_total_bytes: total_bytes,
            system_available_bytes: available_bytes,
        })
    }

    #[cfg(target_os = "linux")]
    fn from_linux() -> io::Result<Self> {
        let status_file = fs::File::open("/proc/self/status")?;
        let reader = io::BufReader::new(status_file);

        let mut rss_kb = 0u64;
        let mut vm_size_kb = 0u64;

        for line in reader.lines() {
            let line = line?;
            if line.starts_with("VmRSS:") {
                rss_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("VmSize:") {
                vm_size_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            }
        }

        // Get system memory from /proc/meminfo
        let meminfo_file = fs::File::open("/proc/meminfo")?;
        let reader = io::BufReader::new(meminfo_file);

        let mut total_kb = 0u64;
        let mut available_kb = 0u64;

        for line in reader.lines() {
            let line = line?;
            if line.starts_with("MemTotal:") {
                total_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if line.starts_with("MemAvailable:") {
                available_kb = line
                    .split_whitespace()
                    .nth(1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            }
        }

        Ok(Self {
            rss_bytes: rss_kb * 1024,
            virtual_bytes: vm_size_kb * 1024,
            system_total_bytes: total_kb * 1024,
            system_available_bytes: available_kb * 1024,
        })
    }

    /// Format bytes as human-readable string
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_idx = 0;

        while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
            size /= 1024.0;
            unit_idx += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_idx])
    }

    /// Get RSS as human-readable string
    pub fn rss_formatted(&self) -> String {
        Self::format_bytes(self.rss_bytes)
    }

    /// Get virtual memory as human-readable string
    pub fn virtual_formatted(&self) -> String {
        Self::format_bytes(self.virtual_bytes)
    }

    /// Get available system memory as human-readable string
    pub fn available_formatted(&self) -> String {
        Self::format_bytes(self.system_available_bytes)
    }

    /// Get total system memory as human-readable string
    pub fn total_formatted(&self) -> String {
        Self::format_bytes(self.system_total_bytes)
    }

    /// Calculate memory usage percentage
    pub fn usage_percentage(&self) -> f64 {
        if self.system_total_bytes == 0 {
            return 0.0;
        }
        (self.rss_bytes as f64 / self.system_total_bytes as f64) * 100.0
    }

    /// Check if memory usage is safe (below threshold)
    pub fn is_safe(&self, threshold_percentage: f64) -> bool {
        self.usage_percentage() < threshold_percentage
    }

    /// Estimate if we can safely allocate more memory
    pub fn can_allocate(&self, additional_bytes: u64, safety_margin_gb: f64) -> bool {
        let safety_margin_bytes = (safety_margin_gb * 1024.0 * 1024.0 * 1024.0) as u64;
        let required = self.rss_bytes + additional_bytes + safety_margin_bytes;
        required < self.system_total_bytes
    }
}

/// Memory monitor that tracks usage over time
pub struct MemoryMonitor {
    last_check: Option<MemoryInfo>,
    max_rss_bytes: u64,
    threshold_percentage: f64,
}

impl MemoryMonitor {
    /// Create a new memory monitor with a threshold percentage (e.g., 80.0)
    pub fn new(threshold_percentage: f64) -> Self {
        Self {
            last_check: None,
            max_rss_bytes: 0,
            threshold_percentage,
        }
    }

    /// Update and check memory usage
    pub fn check(&mut self) -> io::Result<MemoryInfo> {
        let info = MemoryInfo::current()?;

        if info.rss_bytes > self.max_rss_bytes {
            self.max_rss_bytes = info.rss_bytes;
        }

        self.last_check = Some(info.clone());
        Ok(info)
    }

    /// Check if current memory usage exceeds threshold
    pub fn is_over_threshold(&self) -> bool {
        if let Some(ref info) = self.last_check {
            !info.is_safe(self.threshold_percentage)
        } else {
            false
        }
    }

    /// Get maximum RSS observed
    pub fn max_rss_formatted(&self) -> String {
        MemoryInfo::format_bytes(self.max_rss_bytes)
    }

    /// Print memory report
    pub fn print_report(&self) {
        if let Some(ref info) = self.last_check {
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("Memory Usage Report");
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
            println!("  Process RSS:       {}", info.rss_formatted());
            println!("  Process Virtual:   {}", info.virtual_formatted());
            println!("  Max RSS:           {}", self.max_rss_formatted());
            println!("  System Total:      {}", info.total_formatted());
            println!("  System Available:  {}", info.available_formatted());
            println!("  Usage:             {:.1}%", info.usage_percentage());
            println!("  Threshold:         {:.1}%", self.threshold_percentage);

            if self.is_over_threshold() {
                println!("  Status:            ⚠️  OVER THRESHOLD");
            } else {
                println!("  Status:            ✓ SAFE");
            }
            println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_info() {
        let info = MemoryInfo::current().unwrap();
        assert!(info.rss_bytes > 0);
        assert!(info.system_total_bytes > 0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(MemoryInfo::format_bytes(1024), "1.00 KB");
        assert_eq!(MemoryInfo::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(MemoryInfo::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_memory_monitor() {
        let mut monitor = MemoryMonitor::new(80.0);
        let info = monitor.check().unwrap();
        assert!(info.rss_bytes > 0);
        assert!(monitor.max_rss_bytes > 0);
    }
}
