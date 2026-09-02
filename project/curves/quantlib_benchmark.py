"""QuantLib-based benchmark comparison for the scratch-built Nelson-Siegel curve fit.

Treasury Constant Maturity Treasury (CMT) yields are treated here as
continuously-compounded zero rates (Act/365 Fixed) for the purpose of
building a QuantLib reference curve -- a standard simplification (CMT
yields are technically semi-annual bond-equivalent par yields on
on-the-run/interpolated instruments, not zero rates) that's fine for a
shape-comparison benchmark but not for actual pricing. QuantLib's curve is
built by cubic-spline interpolation directly through the observed points
(`ql.ZeroCurve`), structurally different from this project's parametric
4-parameter Nelson-Siegel fit -- comparing the two shows how much curve
shape the NS parametrization's low dimensionality gives up relative to an
interpolation-only curve that reproduces every quote exactly.
"""

import numpy as np
import QuantLib as ql

from project.curves.nelson_siegel import nelson_siegel_yield


def build_quantlib_zero_curve(tenors, yields, evaluation_date=None):
    """Build a QuantLib ZeroCurve from tenor(years)/yield(decimal) pairs."""
    if evaluation_date is None:
        evaluation_date = ql.Date.todaysDate()
    ql.Settings.instance().evaluationDate = evaluation_date

    dates = [evaluation_date]
    rates = [float(yields[0])]
    for tenor, y in zip(tenors, yields):
        days = max(1, int(round(tenor * 365)))
        dates.append(evaluation_date + ql.Period(days, ql.Days))
        rates.append(float(y))

    day_count = ql.Actual365Fixed()
    curve = ql.ZeroCurve(dates, rates, day_count)
    return curve, day_count, evaluation_date


def quantlib_zero_rates_at(curve, day_count, evaluation_date, maturities):
    """Query the QuantLib curve's continuously-compounded zero rate at each maturity-in-years."""
    rates = []
    for T in maturities:
        d = evaluation_date + ql.Period(max(1, int(round(T * 365))), ql.Days)
        rates.append(curve.zeroRate(d, day_count, ql.Continuous).rate())
    return np.array(rates)


def compare_ns_to_quantlib(tenors, yields, ns_params, maturities, evaluation_date=None) -> dict:
    """Compare the NS-fitted zero curve to a QuantLib interpolated benchmark
    curve built from the same day's raw quotes, at each maturity.

    `ns_params` is [b0, b1, b2, lam]. Returns
    {"ns_zero_rates", "quantlib_zero_rates", "rmse"} (rmse in the same
    decimal units as `yields`).
    """
    curve, day_count, eval_date = build_quantlib_zero_curve(tenors, yields, evaluation_date)
    ql_rates = quantlib_zero_rates_at(curve, day_count, eval_date, maturities)
    ns_rates = nelson_siegel_yield(np.asarray(maturities, dtype=float), *ns_params)
    rmse = float(np.sqrt(np.mean((ns_rates - ql_rates) ** 2)))
    return {"ns_zero_rates": ns_rates, "quantlib_zero_rates": ql_rates, "rmse": rmse}
