library(tidyverse)
library(conflicted)
filter <- dplyr::filter
library(glue)

synthetic_mols <- read_csv("pipeline_example_mol_tbl.csv.gz", col_types = cols(mol_id = col_character()))
synthetic_one_atom <- read_csv("pipeline_example_one_tbl.csv.gz", col_types = cols(
  mol_id = col_character(),
  atom_id = col_character(),
  symbol = col_character(),
  .default = col_double()
)) |>
  select(mol_id, atom_id, symbol)
synthetic_two_atom <- read_csv("pipeline_example_two_tbl.csv.gz", col_types = cols(
  mol_id = col_character(),
  bond_id = col_character(),
  start_atom = col_character(),
  end_atom = col_character(),
  .default = col_character()
)) |>
  select(mol_id, bond_id, start_atom, end_atom)

element_features <- read_csv("element_features.csv",
    col_types = cols(symbol = col_character(), .default = col_double()))

train_stats <- read_csv("train_stats.csv",
    col_types = cols(feature = col_character(),
                     center = col_double(),
                     scale = col_double()))

synthetic_mols |>
    write_csv("featurized_mol_tbl.csv.gz")

synthetic_one_atom |>
  # Join in element features
  left_join(element_features, by = "symbol") |>
  select(-symbol) |>
  # Scale all features to have mean 0 and standard deviation 1
  pivot_longer(-c(mol_id, atom_id), names_to = "feature", values_to = "value") |>
  left_join(train_stats, by = "feature") |>
  mutate(z = (value - center / scale)) |>
  pivot_wider(id_cols = c(mol_id, atom_id), names_from = "feature", values_from = "z") |>
  write_csv("featurized_one_tbl.csv.gz")

synthetic_two_atom |>
  select(mol_id, bond_id, start_atom, end_atom) |>
  write_csv("featurized_two_tbl.csv.gz")
