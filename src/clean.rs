use crate::models::LifeExpectancyRecord;
use csv::{ReaderBuilder, WriterBuilder};
use std::error::Error;

pub fn clean_data(file_path: &str, output_file: &str) -> Result<(), Box<dyn Error>> {
    // Step 1: Load the dataset
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    let mut records: Vec<LifeExpectancyRecord> = Vec::new();

    for result in rdr.deserialize() {
        let record: LifeExpectancyRecord = result?;
        records.push(record);
    }

    // Step 2: Handle missing values
    let mut cleaned_records = Vec::new();
    for mut record in records {
        record.LifeExpectancy = record.LifeExpectancy.or(Some(65.0)); // Example placeholder for Life Expectancy
        record.IncomeResources = record.IncomeResources.or(Some(0.5)); // Default for Income Composition
        record.GDP = record.GDP.or(Some(5000.0)); // Default GDP value

        // Handle missing values for other fields
        record.AdultMortality = record.AdultMortality.or(Some(0.0)); // Default for Adult Mortality
        record.InfantDeaths = record.InfantDeaths.or(Some(0.0)); // Default for Infant Deaths
        record.Schooling = record.Schooling.or(Some(0.0)); // Default for Schooling

        // Add the cleaned record to the vector
        cleaned_records.push(record);
    }

    // Step 3: Write cleaned data to a new CSV file
    let mut wtr = WriterBuilder::new().has_headers(true).from_path(output_file)?;

    for record in cleaned_records {
        wtr.serialize(record)?;
    }

    wtr.flush()?;
    println!("Data cleaned and saved to '{}'.", output_file);

    Ok(())
}

pub fn load_cleaned_data(file_path: &str) -> Result<Vec<LifeExpectancyRecord>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new()
        .has_headers(true)
        .from_path(file_path)?;

    let mut records: Vec<LifeExpectancyRecord> = Vec::new();
    for result in rdr.deserialize() {
        let record: LifeExpectancyRecord = result?;
        records.push(record);
    }

    Ok(records)
}

