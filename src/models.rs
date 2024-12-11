use std::option::Option;
use csv::{ReaderBuilder, WriterBuilder};
use std::collections::HashMap;
use std::error::Error;

use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct LifeExpectancyRecord {
    pub(crate) Country: String,
    pub(crate) Year: u16,
    pub(crate) Status: String,
    #[serde(rename = "Life expectancy")]
    pub(crate) LifeExpectancy: std::option::Option<f64>,
    #[serde(rename = "Income composition of resources")]
    pub(crate) IncomeResources: std::option::Option<f64>,
    pub(crate) GDP: std::option::Option<f64>,
    #[serde(rename = "Adult Mortality")]
    pub(crate) AdultMortality: std::option::Option<f64>,
    #[serde(rename = "infant deaths")]
    pub(crate) InfantDeaths: std::option::Option<f64>,
    pub(crate) Schooling: std::option::Option<f64>,

}

#[derive(Debug, Deserialize, Serialize)]
pub(crate) struct HappinessRecord {
    pub(crate) Country: std::string::String,

    #[serde(rename = "Happiness Rank")]
    pub(crate) rank: std::option::Option<u32>,

    #[serde(rename = "Happiness Score")]
    pub(crate) score: std::option::Option<f64>,

    #[serde(rename = "Standard Error")]
    pub(crate) standard_error: std::option::Option<f32>, // Make this field optional

    #[serde(rename = "Economy (GDP per Capita)")]
    pub(crate) economy: std::option::Option<f32>,

    #[serde(rename = "Health (Life Expectancy)")]
    pub(crate) life_expectancy: std::option::Option<f64>,
}

