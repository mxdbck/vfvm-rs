#![allow(unused)]
use std::cell::RefCell;
use std::time::Duration;

#[derive(Default, Clone)]
pub struct TimingStats {
    pub jacobian_times: Vec<Duration>,
    pub linear_solve_times: Vec<Duration>,
    pub total_time: Duration,
}

impl TimingStats {
    pub fn new() -> Self {
        Self::default()
    }

    #[cfg(feature = "timing")]
    pub fn print_summary(&self) {
        if self.jacobian_times.is_empty() {
            return;
        }

        let total_jacobian: Duration = self.jacobian_times.iter().sum();
        let total_linear: Duration = self.linear_solve_times.iter().sum();

        let accounted = total_jacobian + total_linear;
        let overhead = self.total_time.saturating_sub(accounted);

        // Use the minimum of the two lengths for averaging
        let n = self.jacobian_times.len().min(self.linear_solve_times.len());

        println!("\n{}", "=".repeat(60));
        println!("{:^60}", "SOLVER TIMING SUMMARY");
        println!("{}", "=".repeat(60));
        println!(
            "Total solver time:             {:.3}s",
            self.total_time.as_secs_f64()
        );
        println!("{}", "-".repeat(60));
        println!("Component breakdown:");
        println!(
            "  Jacobian assembly:         {:>9.3}ms  (avg: {:>9.3}ms)",
            total_jacobian.as_secs_f64() * 1000.0,
            total_jacobian.as_secs_f64() * 1000.0 / self.jacobian_times.len() as f64
        );
        println!(
            "  Linear solve:              {:>9.3}s   (avg: {:>9.3}ms)",
            total_linear.as_secs_f64(),
            total_linear.as_secs_f64() * 1000.0 / self.linear_solve_times.len() as f64
        );
        println!("{}", "=".repeat(60));
        println!(
            "Overhead/Other:                {:>9.3}ms",
            overhead.as_secs_f64() * 1000.0
        );
        println!(
            "Iterations:                    {} jacobian, {} lin.solve\n",
            self.jacobian_times.len(),
            self.linear_solve_times.len()
        );
    }

    #[cfg(not(feature = "timing"))]
    pub fn print_summary(&self) {}

    #[cfg(feature = "timing")]
    pub fn print_detailed(&self) {
        // Use minimum length to avoid index out of bounds
        let n = self.jacobian_times.len().min(self.linear_solve_times.len());

        if n == 0 {
            return;
        }

        println!("\n{}", "=".repeat(39));
        println!("{:^39}", "DETAILED ITERATION TIMINGS");
        println!("{}", "=".repeat(39));
        println!(
            "{:^6} | {:^10} | {:^7} | {:^7}",
            "Iter", "Jacobian", "Solve", "Total"
        );
        println!("{}", "-".repeat(39));

        for i in 0..n {
            let iter_total = self.jacobian_times[i] + self.linear_solve_times[i];

            println!(
                "{:^6} | {:>8.1}ms | {:>5.1}ms | {:>5.1}ms",
                i,
                self.jacobian_times[i].as_secs_f64() * 1000.0,
                self.linear_solve_times[i].as_secs_f64() * 1000.0,
                iter_total.as_secs_f64() * 1000.0
            );
        }

        // If there's an extra jacobian computation (convergence check), note it
        if self.jacobian_times.len() > n {
            println!(
                "{:^6} | {:>8.1}ms | {:>7} | {:>5.1}ms",
                n,
                self.jacobian_times[n].as_secs_f64() * 1000.0,
                "(conv)",
                self.jacobian_times[n].as_secs_f64() * 1000.0
            );
        }

        println!();
    }

    #[cfg(not(feature = "timing"))]
    pub fn print_detailed(&self) {}
}

#[cfg(feature = "timing")]
thread_local! {
    static TIMING_STATS: RefCell<TimingStats> = RefCell::new(TimingStats::new());
}

#[cfg(feature = "timing")]
pub fn reset_timing() {
    TIMING_STATS.with(|stats| {
        *stats.borrow_mut() = TimingStats::new();
    });
}

#[cfg(not(feature = "timing"))]
pub fn reset_timing() {}

#[cfg(feature = "timing")]
pub fn record_jacobian<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    TIMING_STATS.with(|stats| {
        stats.borrow_mut().jacobian_times.push(elapsed);
    });
    result
}

#[cfg(not(feature = "timing"))]
pub fn record_jacobian<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}

#[cfg(feature = "timing")]
pub fn record_linear_solve<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    let start = std::time::Instant::now();
    let result = f();
    let elapsed = start.elapsed();
    TIMING_STATS.with(|stats| {
        stats.borrow_mut().linear_solve_times.push(elapsed);
    });
    result
}

#[cfg(not(feature = "timing"))]
pub fn record_linear_solve<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    f()
}

#[cfg(feature = "timing")]
pub fn finalize_timing(total_time: Duration) -> TimingStats {
    TIMING_STATS.with(|stats| {
        let mut s = stats.borrow_mut();
        s.total_time = total_time;
        s.clone()
    })
}

#[cfg(not(feature = "timing"))]
pub fn finalize_timing(_total_time: Duration) -> TimingStats {
    TimingStats::new()
}

#[cfg(feature = "timing")]
pub fn finalize_and_print(total_time: Duration) {
    let stats = finalize_timing(total_time);
    stats.print_summary();
    stats.print_detailed();
}

#[cfg(not(feature = "timing"))]
pub fn finalize_and_print(_total_time: Duration) {}
