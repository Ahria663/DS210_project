use crate::models::LifeExpectancyRecord;
use plotters::prelude::*;
use statrs::statistics::{Data, Distribution, Median};
use std::error::Error;

pub fn perform_eda(records: &[LifeExpectancyRecord]) -> Result<(), Box<dyn Error>> {
    // Step 1: Filter GDP values and country names
    let countries: Vec<String> = records.iter().filter_map(|r| {
        if r.GDP.is_some() {
            Some(r.Country.clone())
        } else {
            None
        }
    }).collect();

    let gdp_per_country: Vec<f64> = records.iter().filter_map(|r| r.GDP).collect();  // GDP per country

    // Step 2: Generate a bar chart for GDP per country
    if !countries.is_empty() && !gdp_per_country.is_empty() {
        generate_gdp_line_plot(&countries, &gdp_per_country, "GDP per Country Distribution")?;
    } else {
        println!("No valid GDP data available for the bar chart.");
    }

    // Step 3: Generate histograms for Adult Mortality and Infant Deaths
    let adult_mortality: Vec<f64> = records.iter().filter_map(|r| r.AdultMortality).collect();
    let infant_deaths: Vec<f64> = records.iter().filter_map(|r| r.InfantDeaths).collect();

    if !adult_mortality.is_empty() && !infant_deaths.is_empty() {
        generate_double_histogram(&adult_mortality, &infant_deaths)?;
    } else {
        println!("No valid data for Adult Mortality or Infant Deaths.");
    }

    // Step 4: Calculate basic statistics for Life Expectancy and GDP per country
    let life_expectancy: Vec<f64> = records
        .iter()
        .filter_map(|r| r.LifeExpectancy)
        .filter(|&v| !v.is_nan())
        .collect();

    if !life_expectancy.is_empty() && !gdp_per_country.is_empty() {
        println!("Life Expectancy vs GDP Statistics:");

        // Compute stats for Life Expectancy
        let life_expectancy_stats = Data::new(life_expectancy.clone());
        println!("Life Expectancy Statistics:");
        println!("Mean: {:?}", life_expectancy_stats.mean());
        println!("Median: {:?}", life_expectancy_stats.median());
        println!("Standard deviation: {:?}", life_expectancy_stats.std_dev());
        println!("Variance: {:?}", life_expectancy_stats.variance());

        // Compute stats for GDP per country
        let gdp_stats = Data::new(gdp_per_country.clone());
        println!("GDP per Country Statistics:");
        println!("Mean: {:?}", gdp_stats.mean());
        println!("Median: {:?}", gdp_stats.median());
        println!("Standard deviation: {:?}", gdp_stats.std_dev());
        println!("Variance: {:?}", gdp_stats.variance());
    }
    Ok(())
}

// Function to generate a bar chart for GDP per country distribution
// fn generate_gdp_bar_chart(countries: &[String], gdp_values: &[f64], title: &str) -> Result<(), Box<dyn Error>> {
//     let root = BitMapBackend::new("gdp_per_country_bar_chart.png", (1200, 800)).into_drawing_area();
//     root.fill(&WHITE)?;
//
//     let mut chart = ChartBuilder::on(&root)
//         .caption(title, ("sans-serif", 20))
//         .margin(10)
//         .x_label_area_size(80)
//         .y_label_area_size(80)
//         .build_cartesian_2d(
//             0..(countries.len() as u32),
//             0..gdp_values.iter().map(|&v| v as u32).max().unwrap_or(0),
//         )?;
//
//     chart.configure_mesh().x_labels(10).y_labels(10).draw()?;
//
//     chart.draw_series(
//         countries.iter().zip(gdp_values.iter()).enumerate().map(|(i, (country, &gdp))| {
//             let x_pos = i as u32;
//             let y_pos = gdp as u32;
//             Rectangle::new(
//                 [(x_pos, 0), (x_pos + 1, y_pos)],
//                 BLUE.filled(),
//             )
//         }),
//     )?;
//
//     for (i, country) in countries.iter().enumerate() {
//         let x_pos = i as u32;
//         chart.draw_series(std::iter::once(Text::new(
//             country.clone(),
//             (x_pos, 0),
//             ("sans-serif", 10).into_font(),
//         )))?;
//     }
//
//     root.present()?;
//     println!("GDP per Country Bar Chart saved to gdp_per_country_bar_chart.png");
//
//     Ok(())
// }

// Function to generate a double histogram for Adult Mortality and Infant Deaths
fn generate_double_histogram(adult_mortality: &[f64], infant_deaths: &[f64]) -> Result<(), Box<dyn Error>> {
    let root = BitMapBackend::new("double_histogram.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let areas = root.split_evenly((2, 1));  // Split into two areas for the histograms

    // Create histograms for Adult Mortality and Infant Deaths
    let mut chart1 = ChartBuilder::on(&areas[0])
        .caption("Adult Mortality Distribution", ("Arial", 20).into_font())
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0u32..100u32, 0u32..100u32)?;  // Discrete x-axis
    chart1.configure_mesh().draw()?;

    let mut chart2 = ChartBuilder::on(&areas[1])
        .caption("Infant Deaths Distribution", ("Arial", 20).into_font())
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(0u32..100u32, 0u32..100u32)?;  // Discrete x-axis
    chart2.configure_mesh().draw()?;

    // Draw Adult Mortality histogram
    let am_color = RGBColor(142, 166, 4);
    let adult_mortality_bins = adult_mortality.iter().map(|&x| x.round() as u32);  // Round to nearest integer for bins
    chart1.draw_series(
        Histogram::vertical(&chart1)
            .style(am_color.filled())
            .data(adult_mortality_bins.map(|x| (x, 1))),
    )?;

    // Draw Infant Deaths histogram
    let id_color = RGBColor(255, 78, 0);
    let infant_deaths_bins = infant_deaths.iter().map(|&x| x.round() as u32);  // Round to nearest integer for bins
    chart2.draw_series(
        Histogram::vertical(&chart2)
            .style(id_color.filled())
            .data(infant_deaths_bins.map(|x| (x, 1))),
    )?;

    root.present()?;
    println!("Double Histogram saved to double_histogram.png");

    Ok(())
}

use plotters::prelude::*;

fn generate_gdp_line_plot(countries: &[String], gdp_values: &[f64], title: &str) -> Result<(), Box<dyn Error>> {
    // Prepare the chart
    let root = BitMapBackend::new("gdp_per_country_line_plot.png", (1200, 800)).into_drawing_area();
    root.fill(&WHITE)?;

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 20))
        .margin(10)
        .x_label_area_size(80)
        .y_label_area_size(80)
        .build_cartesian_2d(0..(countries.len() as u32), 0f64..gdp_values.iter().cloned().fold(0.0/0.0, f64::max))?;

    chart.configure_mesh().x_labels(10).y_labels(10).draw()?;

    // Draw the line graph by connecting each point with a line
    chart.draw_series(
        LineSeries::new(
            countries.iter().zip(gdp_values.iter()).enumerate().map(|(i, (_, &gdp))| {
                let x_pos = i as u32; // X-axis position based on country index
                let y_pos = gdp; // Y-axis position based on GDP value
                (x_pos, y_pos) // Create tuple of coordinates for the line
            }),
            &BLUE, // Line color
        )
    )?;

    // Add country names as annotations below the points (optional)
    for (i, country) in countries.iter().enumerate() {
        let x_pos = i as u32;
        let y_pos = gdp_values[i];

        // Create the text element to annotate the points
        let text = Text::new(
            country.clone(),
            (x_pos, y_pos - 0.5),  // Place text slightly below the point
            ("sans-serif", 10).into_font(),
        );

        // Draw the text annotation
        chart.draw_series(std::iter::once(text))?;
    }

    root.present()?;
    println!("GDP per Country Line Plot saved to gdp_per_country_line_plot.png");

    Ok(())
}



