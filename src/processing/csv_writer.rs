use std::fs::File;
use std::io::{self, Write};
use std::path::Path;

/// Write data to CSV file with headers
pub fn write_csv<P: AsRef<Path>>(path: P, headers: &[&str], data: &[Vec<f64>]) -> io::Result<()> {
    if !headers.is_empty() && !data.is_empty() && headers.len() != data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "Headers count ({}) doesn't match data columns ({})",
                headers.len(),
                data.len()
            ),
        ));
    }

    let mut file = File::create(path)?;

    writeln!(file, "{}", headers.join(","))?;

    let n_rows = data.iter().map(|col| col.len()).max().unwrap_or(0);

    for i in 0..n_rows {
        let row: Vec<String> = data
            .iter()
            .map(|col| {
                if i < col.len() {
                    format!("{:.15e}", col[i])
                } else {
                    String::new()
                }
            })
            .collect();
        writeln!(file, "{}", row.join(","))?;
    }

    Ok(())
}

/// Write a single column of data with a header
pub fn write_single_column<P: AsRef<Path>>(path: P, header: &str, data: &[f64]) -> io::Result<()> {
    write_csv(path, &[header], &[data.to_vec()])
}

/// Write x-y data pairs
pub fn write_xy<P: AsRef<Path>>(
    path: P,
    x_header: &str,
    y_header: &str,
    x_data: &[f64],
    y_data: &[f64],
) -> io::Result<()> {
    if x_data.len() != y_data.len() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "X and Y data lengths don't match ({} vs {})",
                x_data.len(),
                y_data.len()
            ),
        ));
    }
    write_csv(
        path,
        &[x_header, y_header],
        &[x_data.to_vec(), y_data.to_vec()],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_write_csv() {
        let path = "test_output.csv";
        let headers = &["x", "y", "z"];
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];

        write_csv(path, headers, &data).unwrap();

        let content = fs::read_to_string(path).unwrap();
        assert!(content.contains("x,y,z"));

        fs::remove_file(path).ok();
    }
}
