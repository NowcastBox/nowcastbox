# Export the datasets bundled with the R package "nowcasting" (USGDP, NYFED, BRGDP) as
# test INPUTS (tests/reference_validation/fixtures/inputs/). They are used only inside
# tests/reference_validation and are never shipped with nowcastbox.
#
# Run from the repository root:  Rscript scripts/reference_fixtures/export_inputs.R

source(file.path("scripts", "reference_fixtures", "_common.R"))

data(USGDP)
data(NYFED)
data(BRGDP)

# --- USGDP (Giannone, Reichlin & Small, 2008 replication files) ---------------------
write_ts_csv(USGDP$base, file.path(DIR_INPUTS, "usgdp_base.csv.gz"), digits = 15)
write.csv(
  data.frame(name = colnames(USGDP$base), transformation = USGDP$legend$Transformation),
  file.path(DIR_INPUTS, "usgdp_legend.csv"), row.names = FALSE, quote = FALSE
)

# --- NYFED (FRBNY Staff Nowcast example panel) ---------------------------------------
write_ts_csv(NYFED$base, file.path(DIR_INPUTS, "nyfed_base.csv.gz"), digits = 15)
blocks <- NYFED$blocks$blocks
colnames(blocks) <- as.character(NYFED$blocks$BlocksNames)
legend <- data.frame(
  name = NYFED$legend$SeriesID,
  frequency = NYFED$legend$Frequency,
  transformation = NYFED$legend$Transformation,
  delay = NYFED$legend$delay,
  stringsAsFactors = FALSE
)
legend <- cbind(legend, as.data.frame(blocks))
write.csv(legend, file.path(DIR_INPUTS, "nyfed_legend.csv"), row.names = FALSE, quote = FALSE)

# --- BRGDP (Brazilian real-time panel of the package) --------------------------------
write_ts_csv(BRGDP$base, file.path(DIR_INPUTS, "brgdp_base.csv.gz"), digits = 15)
write.csv(
  data.frame(name = colnames(BRGDP$base), transformation = BRGDP$trans, delay = BRGDP$delay),
  file.path(DIR_INPUTS, "brgdp_legend.csv"), row.names = FALSE, quote = FALSE
)
write_ts_csv(BRGDP$GDP, file.path(DIR_INPUTS, "brgdp_gdp.csv"), digits = 15)

write_json_fixture(r_session_info(), "session.json")
message("inputs written to ", DIR_INPUTS)
