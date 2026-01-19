def energy_kwh(series, sample_period):
    return series.sum() * sample_period / 3600

def generate_report(preds, residual, sample_period):
    report = {}

    for k, v in preds.items():
        report[k] = energy_kwh(v, sample_period)

    report["others"] = energy_kwh(residual, sample_period)
    report["total"] = sum(report.values())

    return report
