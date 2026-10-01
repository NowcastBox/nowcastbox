# Reference outputs of ICfactors (Bai & Ng, 2002) and ICshocks (Bai & Ng, 2007) of the
# R package "nowcasting" (black box).
#
# Panels: the balanced part of Bpanel(USGDP) (the GRS 2008 panel), of Bpanel(BRGDP) and
# the predictors of the shared simulated panel. ICshocks returns only q*, so the
# threshold values of m at which q* changes are located by bisection (they equal
# D_k * bound denominator and reveal the statistic and the bound actually used).
#
# Run from the repository root:  Rscript scripts/reference_fixtures/r_selection.R

source(file.path("scripts", "reference_fixtures", "_common.R"))

quiet <- function(expr) suppressWarnings(suppressMessages(expr))
timings <- list()

data(USGDP)
data(BRGDP)
gdp_position <- which(colnames(USGDP$base) == "RGDPGR")
us <- quiet(Bpanel(USGDP$base[, -gdp_position], USGDP$legend$Transformation[-gdp_position]))
us <- us[stats::complete.cases(us), ]
br <- quiet(Bpanel(BRGDP$base, BRGDP$trans, h = 0))
br <- br[stats::complete.cases(br), ]
sim <- read_monthly_csv(file.path(DIR_INPUTS, "simulated.csv"))
sim <- sim[stats::complete.cases(sim[, colnames(sim) != "gdp"]), colnames(sim) != "gdp"]

panels <- list(usgdp = us, brgdp = br, simulated = sim)
ic <- list()
for (nm in names(panels)) {
  x <- panels[[nm]]
  for (rmax in c(8, 15)) {
    for (type in 1:3) {
      t <- timed(quiet(ICfactors(x, rmax = rmax, type = type)), 3)
      key <- sprintf("%s_rmax%d_type%d", nm, rmax, type)
      ic[[key]] <- list(panel = nm, rmax = rmax, type = type, n_rows = nrow(x),
                        n_cols = ncol(x), r_star = t$value$r_star, IC = t$value$IC)
      if (rmax == 15 && type == 2) timings[[paste0("icfactors_", nm)]] <- t$seconds
    }
  }
}
write_json_fixture(ic, "icfactors.json")

grid <- list()
settings <- list(c(0.1, 1), c(0.2, 2), c(0.05, 0.5), c(0.3, 1.5))
for (nm in names(panels)) {
  x <- panels[[nm]]
  for (s in settings) {
    for (r in 2:6) {
      for (p in 1:3) {
        q <- quiet(ICshocks(x, r = r, p = p, delta = s[1], m = s[2])$q_star)
        grid[[length(grid) + 1]] <- list(panel = nm, r = r, p = p, delta = s[1], m = s[2],
                                         q_star = q)
      }
    }
  }
}
t <- timed(quiet(ICshocks(us, r = 4, p = 2)), 3)
timings$icshocks_usgdp <- t$seconds
write_json_fixture(grid, "icshocks_grid.json")

# thresholds: smallest m with q* <= target (bisection on log m)
threshold <- function(x, r, p, delta, target) {
  qf <- function(m) quiet(ICshocks(x, r = r, p = p, delta = delta, m = m)$q_star)
  lo <- 1e-8
  hi <- 1e4
  if (qf(hi) > target) return(NA_real_)
  for (i in 1:70) {
    mid <- sqrt(lo * hi)
    if (qf(mid) <= target) hi <- mid else lo <- mid
  }
  hi
}
thr <- list()
for (case in list(list("usgdp", 4, 1), list("usgdp", 3, 2), list("simulated", 4, 1),
                  list("brgdp", 4, 2))) {
  x <- panels[[case[[1]]]]
  for (delta in c(0.05, 0.1, 0.3, 0.45)) {
    for (target in seq_len(case[[2]] - 1)) {
      thr[[length(thr) + 1]] <- list(panel = case[[1]], r = case[[2]], p = case[[3]],
                                     delta = delta, target = target,
                                     m = threshold(x, case[[2]], case[[3]], delta, target))
    }
  }
}
write_json_fixture(thr, "icshocks_thresholds.json")

record_timings(timings)
message("selection fixtures written to ", DIR_R)
