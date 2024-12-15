use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use csv::Reader;
use ndarray::Array2;
use ordered_float::NotNan;
use plotters::backend::BitMapBackend;
use plotters::chart::ChartBuilder;
use plotters::element::{Circle, PathElement, Rectangle};
use plotters::prelude::*;

// EDA + Statistics
pub(crate) fn find_top_countries(file_path: &str, country_column: usize, year_column: usize, life_expectancy_column: usize) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;

    let mut year_data: HashMap<String, BinaryHeap<Reverse<(NotNan<f64>, String)>>> = HashMap::new();

    for result in reader.records() {
        let record = result?;
        let country = record.get(country_column).unwrap_or("").to_string();
        let year = record.get(year_column).unwrap_or("").to_string();
        let life_expectancy = record
            .get(life_expectancy_column)
            .unwrap_or("0")
            .parse::<f64>()
            .ok()
            .and_then(|val| NotNan::new(val).ok())
            .unwrap_or_else(|| NotNan::new(0.0).unwrap());

        year_data
            .entry(year.clone())
            .or_insert_with(BinaryHeap::new)
            .push(Reverse((life_expectancy, country)));
    }

    for (year, mut heap) in year_data {
        println!("Top 5 countries in year {}:", year);
        let mut top_countries = Vec::new();
        while let Some(Reverse((life_expectancy, country))) = heap.pop() {
            top_countries.push((country, life_expectancy));
            if top_countries.len() == 5 {
                break;
            }
        }

        for (country, life_expectancy) in top_countries {
            println!("{}: {:.2}", country, life_expectancy);
        }
        println!();
    }

    Ok(())
}

// Calculate the correlation between two variables
pub(crate) fn create_correlation_heatmap(
    file_path: &str,
    output_file: &str,
    exclude_columns: &[usize], // Columns to exclude (e.g., Year, Country)
    feature_names: &[String],  // Names of all columns (for labeling the heatmap)
) -> Result<(), Box<dyn Error>> {
    // Load CSV data
    let mut reader = csv::Reader::from_path(file_path)?;
    // let headers = reader.headers()?.clone();

    // Parse the data into a matrix
    let mut data_matrix: Vec<Vec<f64>> = Vec::new();
    for record in reader.records() {
        let record = record?;
        let row: Vec<f64> = record
            .iter()
            .enumerate()
            .filter(|(i, _)| !exclude_columns.contains(i)) // Exclude specific columns
            .map(|(_, value)| value.parse::<f64>().unwrap_or(0.0)) // Parse as f64, default to 0.0
            .collect();
        data_matrix.push(row);
    }

    let data = Array2::from_shape_vec(
        (data_matrix.len(), data_matrix[0].len()),
        data_matrix.into_iter().flatten().collect(),
    )?;

    let (_, cols) = data.dim();
    if cols == 0 {
        return Err("No columns to process".into());
    }

    // Calculate the correlation matrix
    let mut correlation_matrix = Array2::zeros((cols, cols));
    for i in 0..cols {
        for j in 0..cols {
            let col_i = data.column(i);
            let col_j = data.column(j);
            let correlation = calculate_correlation(&col_i.view(), &col_j.view()).unwrap_or(0.0);
            correlation_matrix[(i, j)] = correlation;
        }
    }

    let root = BitMapBackend::new(output_file, (1024, 1024)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Feature Correlation Heatmap", ("sans-serif", 30))
        .margin(5)
        .x_label_area_size(60)
        .y_label_area_size(60)
        .build_cartesian_2d(0..cols as u32, 0..cols as u32)?;

    chart
        .configure_mesh()
        .disable_mesh()
        .x_labels(cols)
        .y_labels(cols)
        .x_desc("Features")
        .y_desc("Features")
        .label_style(("sans-serif", 15))
        .axis_desc_style(("sans-serif", 20))
        .draw()?;

    // Add labels for axes (feature names)
    chart.configure_mesh().x_label_formatter(&|x| {
        feature_names
            .get(*x as usize)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    });
    chart.configure_mesh().y_label_formatter(&|y| {
        feature_names
            .get(*y as usize)
            .cloned()
            .unwrap_or_else(|| "Unknown".to_string())
    });

    // Draw heatmap rectangles
    for i in 0..cols {
        for j in 0..cols {
            let value = correlation_matrix[(i, j)];
            let color = if value >= 0.0 {
                RGBColor((255.0 * (1.0 - value)) as u8, (255.0 * value) as u8, 0)
            } else {
                RGBColor(0, (255.0 * (1.0 + value)) as u8, (255.0 * (-value)) as u8)
            };
            chart.draw_series(std::iter::once(Rectangle::new(
                [
                    (j as u32, cols as u32 - i as u32 - 1),
                    ((j + 1) as u32, cols as u32 - i as u32),
                ],
                color.filled(),
            )))?;
        }
    }

    println!("Heatmap saved to {}", output_file);
    Ok(())
}

// Helper function to calculate correlation
fn calculate_correlation(x: &ndarray::ArrayView1<f64>, y: &ndarray::ArrayView1<f64>) -> Option<f64> {
    let x_mean = x.mean()?;
    let y_mean = y.mean()?;
    let numerator = x.iter().zip(y.iter()).map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean)).sum::<f64>();
    let denominator_x = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum::<f64>().sqrt();
    let denominator_y = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<f64>().sqrt();
    if denominator_x > 0.0 && denominator_y > 0.0 {
        Some(numerator / (denominator_x * denominator_y))
    } else {
        None
    }
}

pub(crate) fn create_scatter_plot(file_path: &str, output_file: &str, income_comp_column: usize, schooling_column: usize) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;

    let mut income = Vec::new();
    let mut schoolings = Vec::new();

    for result in reader.records() {
        let record = result?;

        if let (Some(income_value), Some(schooling)) = (
            record.get(income_comp_column),
            record.get(schooling_column),
        ) {
            if let (Ok(income_value), Ok(schooling)) = (
                income_value.parse::<f64>(),
                schooling.parse::<f64>(),
            ) {
                income.push(income_value);
                schoolings.push(schooling);
            }
        }
    }

    let root = BitMapBackend::new(output_file, (1024, 768)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Income vs. Schooling Rates", ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
            0.0..income.iter().cloned().fold(f64::NAN, f64::max),
            0.0..schoolings.iter().cloned().fold(f64::NAN, f64::max),
        )?;

    chart.configure_mesh()
        .x_desc("Income")
        .y_desc("Schooling Rates")
        .draw()?;

    chart.draw_series(
        income.iter().zip(schoolings.iter()).map(|(&x, &y)| {
            Circle::new((x, y), 3, RGBAColor(190, 86, 131, 0.5).filled())
        }),
    )?;

    println!("Scatter plot saved to {}", output_file);
    Ok(())
}

// Calculate average life expectancy developing vs developed countries
pub(crate) fn calculate_average_life_expectancy(
    file_path: &str,
    _country_column: usize,
    status_column: usize,
    life_expectancy_column: usize,
) -> Result<(), Box<dyn Error>> {
    let mut reader = Reader::from_path(file_path)?;

    let mut totals: HashMap<String, (f64, usize)> = HashMap::new();

    for result in reader.records() {
        let record = result?;
        let country_status = record.get(status_column).unwrap_or("").to_string();
        let life_expectancy = record
            .get(life_expectancy_column)
            .unwrap_or("0")
            .parse::<f64>()
            .unwrap_or(0.0);

        if !country_status.is_empty() {
            let entry = totals.entry(country_status).or_insert((0.0, 0));
            entry.0 += life_expectancy;
            entry.1 += 1;
        }
    }

    for (status, (total_life_expectancy, count)) in totals {
        let average = total_life_expectancy / count as f64;
        println!(
            "Average life expectancy for {} countries: {:.2}",
            status, average
        );
    }

    Ok(())
}

pub(crate) fn create_developed_vs_developing_plot(
    file_path: &str,
    output_file: &str,
    feature_column: usize,
    year_column: usize,
    status_column: usize,
) -> Result<(), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(file_path)?;

    let mut data: HashMap<(String, String), Vec<f64>> = HashMap::new();

    for record in reader.records() {
        let record = record?;
        let year = record.get(year_column).unwrap_or("").to_string();
        let status = record.get(status_column).unwrap_or("").to_string();
        let feature_value: f64 = record
            .get(feature_column)
            .unwrap_or("0")
            .parse()
            .unwrap_or(0.0);

        data.entry((year, status))
            .or_insert_with(Vec::new)
            .push(feature_value);
    }

    let mut averages: HashMap<(String, String), f64> = HashMap::new();
    for ((year, status), values) in data {
        let avg = values.iter().copied().sum::<f64>() / values.len() as f64;
        averages.insert((year.clone(), status.clone()), avg);
    }

    let mut years: Vec<String> = averages.keys().map(|(year, _)| year.clone()).collect();
    years.sort();
    let mut developed = Vec::new();
    let mut developing = Vec::new();

    for year in &years {
        developed.push(averages.get(&(year.clone(), "Developed".to_string())).copied().unwrap_or(0.0));
        developing.push(averages.get(&(year.clone(), "Developing".to_string())).copied().unwrap_or(0.0));
    }

    let root = BitMapBackend::new(output_file, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Developed vs Developing Adult Mortality Averages per Year ", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0..years.len() as u32, 0.0..250.0)?;

    chart.configure_mesh()
        .x_labels(years.len())
        .y_desc("Adult Mortality Averages ")
        .x_desc("Years")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .x_label_formatter(&|x| years.get(*x as usize).unwrap_or(&"".to_string()).clone())
        .draw()?;

    chart.draw_series(LineSeries::new(
        (0..developed.len()).map(|x| x as u32).zip(developed.iter().copied()),
        &RED,
    ))?
        .label("Developed")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart.draw_series(LineSeries::new(
        (0..developing.len()).map(|x| x as u32).zip(developing.iter().copied()),
        &BLUE,
    ))?
        .label("Developing")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
// same code as the one above, differences in the chart size, Y-axis view
pub(crate) fn create_developed_vs_developing_plot_infant(
    file_path: &str,
    output_file: &str,
    feature_column: usize,
    year_column: usize,
    status_column: usize,
) -> Result<(), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(file_path)?;

    let mut data: HashMap<(String, String), Vec<f64>> = HashMap::new();

    for record in reader.records() {
        let record = record?;
        let year = record.get(year_column).unwrap_or("").to_string();
        let status = record.get(status_column).unwrap_or("").to_string();
        let feature_value: f64 = record
            .get(feature_column)
            .unwrap_or("0")
            .parse()
            .unwrap_or(0.0);

        data.entry((year, status))
            .or_insert_with(Vec::new)
            .push(feature_value);
    }

    let mut averages: HashMap<(String, String), f64> = HashMap::new();
    for ((year, status), values) in data {
        let avg = values.iter().copied().sum::<f64>() / values.len() as f64;
        averages.insert((year.clone(), status.clone()), avg);
    }

    let mut years: Vec<String> = averages.keys().map(|(year, _)| year.clone()).collect();
    years.sort();
    let mut developed = Vec::new();
    let mut developing = Vec::new();

    for year in &years {
        developed.push(averages.get(&(year.clone(), "Developed".to_string())).copied().unwrap_or(0.0));
        developing.push(averages.get(&(year.clone(), "Developing".to_string())).copied().unwrap_or(0.0));
    }

    let root = BitMapBackend::new(output_file, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption("Developed vs Developing Infant Mortality Averages per Year ", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0..years.len() as u32, 0.0..50.0)?;

    chart.configure_mesh()
        .x_labels(years.len())
        .y_desc("Infant Mortality Averages ")
        .x_desc("Years")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .x_label_formatter(&|x| years.get(*x as usize).unwrap_or(&"".to_string()).clone())
        .draw()?;

    chart.draw_series(LineSeries::new(
        (0..developed.len()).map(|x| x as u32).zip(developed.iter().copied()),
        &RED,
    ))?
        .label("Developed")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &RED));

    chart.draw_series(LineSeries::new(
        (0..developing.len()).map(|x| x as u32).zip(developing.iter().copied()),
        &BLUE,
    ))?
        .label("Developing")
        .legend(|(x, y)| PathElement::new([(x, y), (x + 20, y)], &BLUE));

    chart
        .configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}

pub(crate) fn create_features_comparison_bar_plot(
    file_path: &str,
    output_file: &str,
    feature_columns: &[usize],
    _year_column: usize,
    status_column: usize,
    feature_names: &[&str],
) -> Result<(), Box<dyn Error>> {
    let mut reader = csv::Reader::from_path(file_path)?;

    let mut data: HashMap<(String, String), Vec<f64>> = HashMap::new();

    for record in reader.records() {
        let record = record?;
        let status = record.get(status_column).unwrap_or("").to_string();

        for (&col, &feature_name) in feature_columns.iter().zip(feature_names.iter()) {
            let feature_value: f64 = record
                .get(col)
                .unwrap_or("0")
                .parse()
                .unwrap_or(0.0);

            data.entry((feature_name.to_string(), status.clone()))
                .or_insert_with(Vec::new)
                .push(feature_value);
        }
    }

    let averages: HashMap<(String, String), f64> = data
        .into_iter()
        .map(|((feature, status), values)| {
            let avg = values.iter().copied().sum::<f64>() / values.len() as f64;
            ((feature, status), avg)
        })
        .collect();

    let developed_averages: Vec<f64> = feature_names
        .iter()
        .map(|&name| *averages.get(&(name.to_string(), "Developed".to_string())).unwrap_or(&0.0))
        .collect();

    let developing_averages: Vec<f64> = feature_names
        .iter()
        .map(|&name| *averages.get(&(name.to_string(), "Developing".to_string())).unwrap_or(&0.0))
        .collect();

    let root = BitMapBackend::new(output_file, (1280, 720)).into_drawing_area();
    root.fill(&WHITE)?;

    let max_avg = developed_averages
        .iter()
        .chain(developing_averages.iter())
        .cloned()
        .fold(0.0 / 0.0, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption("Comparison of Features Between Developed and Developing Countries", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .build_cartesian_2d(0..(feature_names.len() as i32 * 2), 0.0..(max_avg * 1.2))?;

    chart
        .configure_mesh()
        .x_labels(feature_names.len() * 2)
        .y_desc("Average")
        .x_desc("Features")
        .axis_desc_style(("sans-serif", 20))
        .label_style(("sans-serif", 15))
        .x_label_formatter(&|x| {
            let index = (*x as usize) / 2;
            feature_names.get(index).unwrap_or(&"").to_string()
        })
        .draw()?;

    // Plot Developed
    chart.draw_series(developed_averages.iter().enumerate().map(|(i, avg)| {
        Rectangle::new(
            [(i as i32 * 2, 0.0), (i as i32 * 2 + 1, *avg)],
            ShapeStyle {
                color: RGBAColor(190, 86, 131, 1f64), // #A5CBC3 for Developing
                filled: true,
                stroke_width: 0,
            },
        )
    }))?
        .label("Developed")
        .legend(|(x, y)| Rectangle::new(
            [(x, y - 5), (x + 10, y + 5)],
            ShapeStyle {
                color: RGBAColor(190, 86, 131, 1f64), // #A5CBC3 for Developing
                filled: true,
                stroke_width: 0,
            },
        ));

    // Plot Developing
    chart.draw_series(developing_averages.iter().enumerate().map(|(i, avg)| {
        Rectangle::new(
            [(i as i32 * 2 + 1, 0.0), (i as i32 * 2 + 2, *avg)],
            ShapeStyle {
                color: RGBAColor(110, 48, 75, 1f64),
                filled: true,
                stroke_width: 0,
            },
        )
    }))?
        .label("Developing")
        .legend(|(x, y)| Rectangle::new(
            [(x, y - 5), (x + 10, y + 5)],
            ShapeStyle {
                color: RGBAColor(110, 48, 75, 1f64),
                filled: true,
                stroke_width: 0,
            },));

    chart
        .configure_series_labels()
        .label_font(("sans-serif", 15))
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    Ok(())
}
