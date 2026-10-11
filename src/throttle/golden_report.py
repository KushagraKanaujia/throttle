#!/usr/bin/env python3
"""Generate HTML report from golden protocol run artifacts."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any


def generate_html_report(
    golden_json_path: Path,
    position_reports: List[Path],
    output_path: Path,
    gpu_hourly_rate: Optional[float] = None,
    operator_name: str = "Throttle",
    operator_email: str = "kush@throttle-pro.com",
) -> None:
    """
    Generate self-contained HTML report from golden protocol artifacts.

    Args:
        golden_json_path: Path to golden.json comparison result
        position_reports: List of paths to B1, C1, B2, C2, B3, C3 position reports
        output_path: Where to write the HTML report
        gpu_hourly_rate: GPU cost in USD/hour (optional, omits cost if not provided)
        operator_name: Name of person who ran the benchmark
        operator_email: Contact email for questions
    """
    # Load golden comparison
    with open(golden_json_path) as f:
        golden = json.load(f)

    # Load position reports
    positions = {}
    for path in position_reports:
        pos_name = path.stem  # B1, B2, B3, C1, C2, C3
        with open(path) as f:
            positions[pos_name] = json.load(f)

    # Extract key data
    decision_eligible = golden.get('decision_eligible', False)
    eligibility_reasons = golden.get('eligibility_reasons', [])

    # Get throughput delta
    throughput_ci = None
    for cond in golden.get('conditions', []):
        if 'throughput_delta_percent_ci' in cond:
            throughput_ci = cond['throughput_delta_percent_ci']
            break

    # Determine recommendation
    overall_outcome = golden.get('overall_outcome', 'unknown')
    if overall_outcome == 'candidate_higher_throughput':
        recommendation = "YES"
        rec_action = "increase"
    elif overall_outcome == 'baseline_higher_throughput':
        recommendation = "NO"
        rec_action = "decrease"
    else:
        recommendation = "INCONCLUSIVE"
        rec_action = "change status unclear for"

    # Extract config details from first baseline and candidate
    b1 = positions.get('B1', {})
    c1 = positions.get('C1', {})

    # Get manifests
    b1_manifest = b1.get('manifest', {})
    c1_manifest = c1.get('manifest', {})

    # Model info
    model_name = b1_manifest.get('model', {}).get('id', 'Unknown model')
    model_revision_full = b1_manifest.get('model', {}).get('immutable_revision', 'Unknown')
    model_revision = model_revision_full[:10] if model_revision_full != 'Unknown' else 'Unknown'

    # GPU info
    gpu_desc = b1_manifest.get('runtime', {}).get('gpu', 'Unknown GPU')
    driver_version = b1_manifest.get('runtime', {}).get('driver_version', 'Unknown')
    cuda_version = b1_manifest.get('runtime', {}).get('cuda_version', 'Unknown')

    # Backend info
    engine_version = b1_manifest.get('engine', {}).get('server_version', 'Unknown')
    engine_name = 'vLLM' if engine_version != 'Unknown' else 'Unknown backend'

    # Extract PyTorch version from image_digest
    image_digest = b1_manifest.get('runtime', {}).get('image_digest', '')
    pytorch_version = 'Unknown'
    if 'pytorch' in image_digest.lower():
        pytorch_version = 'included in image'  # Can't extract specific version from digest

    # Parameter changed
    b1_flags = b1_manifest.get('engine', {}).get('effective_flags', {})
    c1_flags = c1_manifest.get('engine', {}).get('effective_flags', {})

    # Find differing flags
    changed_params = {}
    for key in set(list(b1_flags.keys()) + list(c1_flags.keys())):
        b_val = b1_flags.get(key)
        c_val = c1_flags.get(key)
        if b_val != c_val:
            changed_params[key] = (b_val, c_val)

    # Workload details - get from first position
    total_requests = 0
    total_blocks = 0
    requests_per_position = 0

    # Get from first condition of first position
    if b1.get('conditions'):
        first_cond = b1['conditions'][0]
        requests_per_position = first_cond.get('request_counts', {}).get('attempted', 0)
        total_blocks = len(first_cond.get('blocks', []))

    # Total requests = requests_per_position * 6 positions
    total_requests = requests_per_position * 6
    total_blocks = total_blocks * 6  # 3 blocks per position * 6 positions

    # Get concurrency from condition value
    concurrency = "Unknown"
    if b1.get('conditions'):
        first_cond = b1['conditions'][0]
        concurrency = str(first_cond.get('condition', {}).get('value', 'Unknown'))

    # Calculate cost if GPU rate provided
    cost_data = None
    if gpu_hourly_rate is not None:
        cost_data = calculate_cost_from_positions(positions, gpu_hourly_rate)

    # Get tool versions
    run_version = golden.get('tool_version', 'Unknown')
    report_version = get_current_version()

    # Add fail-fast validation for required fields
    required_fields = {
        'model_name': model_name,
        'model_revision': model_revision,
        'gpu_desc': gpu_desc,
        'driver_version': driver_version,
        'cuda_version': cuda_version,
        'engine_version': engine_version,
        'pytorch_version': pytorch_version,
        'concurrency': concurrency,
    }

    missing_fields = [name for name, value in required_fields.items() if value == 'Unknown']
    if missing_fields:
        raise ValueError(
            f"Cannot generate report: required manifest fields are missing: {', '.join(missing_fields)}. "
            f"Check that position reports contain complete manifest data."
        )

    # Generate HTML
    html = generate_html(
        decision_eligible=decision_eligible,
        eligibility_reasons=eligibility_reasons,
        recommendation=recommendation,
        rec_action=rec_action,
        changed_params=changed_params,
        throughput_ci=throughput_ci,
        cost_data=cost_data,
        model_name=model_name,
        model_revision=model_revision,
        gpu_desc=gpu_desc,
        driver_version=driver_version,
        cuda_version=cuda_version,
        engine_name=engine_name,
        engine_version=engine_version,
        pytorch_version=pytorch_version,
        total_requests=total_requests,
        requests_per_position=requests_per_position,
        total_blocks=total_blocks,
        concurrency=concurrency,
        positions=positions,
        run_version=run_version,
        report_version=report_version,
        operator_name=operator_name,
        operator_email=operator_email,
        gpu_hourly_rate=gpu_hourly_rate,
    )

    # Write output
    with open(output_path, 'w') as f:
        f.write(html)


def calculate_cost_from_positions(positions: Dict[str, Any], gpu_hourly_rate: float) -> Dict[str, Any]:
    """Calculate cost metrics from position reports."""
    baseline_positions = ['B1', 'B2', 'B3']
    candidate_positions = ['C1', 'C2', 'C3']

    def calc_for_variant(pos_names):
        total_tokens = 0
        total_wall_seconds = 0

        for pos_name in pos_names:
            pos = positions.get(pos_name, {})

            # Get wall clock time
            start = datetime.fromisoformat(pos.get('started_at', ''))
            end = datetime.fromisoformat(pos.get('completed_at', ''))
            wall_seconds = (end - start).total_seconds()
            total_wall_seconds += wall_seconds

            # Get total tokens
            for cond in pos.get('conditions', []):
                for block in cond.get('blocks', []):
                    total_tokens += block.get('diagnostic_metrics', {}).get('completion_tokens', 0)

        # Calculate cost per million tokens
        gpu_cost = (total_wall_seconds / 3600) * gpu_hourly_rate
        cost_per_million = (gpu_cost / total_tokens) * 1_000_000 if total_tokens > 0 else 0

        return {
            'total_tokens': total_tokens,
            'wall_seconds': total_wall_seconds,
            'gpu_cost_usd': gpu_cost,
            'cost_per_million_tokens': cost_per_million,
        }

    baseline = calc_for_variant(baseline_positions)
    candidate = calc_for_variant(candidate_positions)

    return {
        'baseline': baseline,
        'candidate': candidate,
    }


def get_current_version() -> str:
    """Get current throttle version."""
    try:
        import importlib.metadata
        return importlib.metadata.version('throttle-pro')
    except Exception:
        return 'Unknown'


def generate_html(
    decision_eligible: bool,
    eligibility_reasons: List[str],
    recommendation: str,
    rec_action: str,
    changed_params: Dict[str, tuple],
    throughput_ci: Optional[Dict],
    cost_data: Optional[Dict],
    model_name: str,
    model_revision: str,
    gpu_desc: str,
    driver_version: str,
    cuda_version: str,
    engine_name: str,
    engine_version: str,
    pytorch_version: str,
    total_requests: int,
    requests_per_position: int,
    total_blocks: int,
    concurrency: str,
    positions: Dict[str, Any],
    run_version: str,
    report_version: str,
    operator_name: str,
    operator_email: str,
    gpu_hourly_rate: Optional[float],
) -> str:
    """Generate the HTML report content."""

    # Build parameter change summary
    param_summary = ""
    for param, (baseline_val, candidate_val) in changed_params.items():
        param_display = param.replace('_', ' ').title()
        param_summary += f"""
        <tr>
            <td><strong>{param_display}</strong></td>
            <td style="text-align: center;">{baseline_val}</td>
            <td style="text-align: center;">{candidate_val}</td>
        </tr>
        """

    # Non-eligible warning banner
    warning_banner = ""
    if not decision_eligible:
        reasons_html = "<br>".join([f"• {r}" for r in eligibility_reasons]) if eligibility_reasons else "See position reports for details"
        warning_banner = f"""
        <div class="warning-banner">
            <h2>⚠ NOT DECISION-ELIGIBLE</h2>
            <p>
                This run did not pass decision gates. No recommendation can be made.
            </p>
            <p><strong>Reasons:</strong></p>
            <p>{reasons_html}</p>
            <p>
                Contact {operator_name} ({operator_email}) for interpretation.
            </p>
        </div>
        """

    # Recommendation section (only if decision-eligible)
    recommendation_section = ""
    if decision_eligible and changed_params:
        first_param = list(changed_params.keys())[0]
        baseline_val, candidate_val = changed_params[first_param]
        param_display = first_param.replace('_', ' ')

        recommendation_section = f"""
        <div class="section">
            <h2>1. RECOMMENDATION</h2>
            <div class="recommendation-box">
                <p class="recommendation-text">
                    <strong>{recommendation}:</strong> {rec_action.capitalize()} {param_display} from {baseline_val} to {candidate_val}
                </p>
                <p>
                    The candidate configuration achieved higher throughput with 95% statistical confidence.
                </p>
            </div>
        </div>
        """

    # Throughput section
    throughput_section = ""
    if throughput_ci:
        estimate = throughput_ci.get('estimate', 0)
        low = throughput_ci.get('low', 0)
        high = throughput_ci.get('high', 0)
        method = throughput_ci.get('method', 'Unknown method')

        throughput_section = f"""
        <div class="section">
            <h2>2. MEASURED IMPACT</h2>

            <h3>Throughput Delta (output tokens/second):</h3>
            <ul>
                <li><strong>Estimate:</strong> {estimate:+.1f}%</li>
                <li><strong>95% Confidence Interval:</strong> {low:+.1f}% to {high:+.1f}%</li>
                <li><strong>Method:</strong> {method.replace('_', ' ').title()}</li>
            </ul>
        """

    # Cost section (only if GPU rate provided)
    cost_section_html = ""
    if cost_data and gpu_hourly_rate is not None:
        baseline_cost = cost_data['baseline']['cost_per_million_tokens']
        candidate_cost = cost_data['candidate']['cost_per_million_tokens']
        savings_pct = ((baseline_cost - candidate_cost) / baseline_cost * 100) if baseline_cost > 0 else 0

        # Determine rate source
        rate_source = "User-supplied assumption"

        cost_section_html = f"""
            <h3>Cost per Million Tokens:</h3>
            <ul>
                <li><strong>Baseline:</strong> ${baseline_cost:.2f}/M tokens</li>
                <li><strong>Candidate:</strong> ${candidate_cost:.2f}/M tokens</li>
                <li><strong>Savings:</strong> {savings_pct:+.1f}% per million tokens</li>
            </ul>

            <h3>Cost Calculation Details:</h3>
            <p style="font-size: 0.9em; color: #666;">
                <strong>GPU Hourly Rate:</strong> ${gpu_hourly_rate:.2f}/hour ({rate_source})<br>
                <strong>Baseline:</strong> {cost_data['baseline']['wall_seconds']:.1f} seconds × ${gpu_hourly_rate:.2f}/hour ÷ 3600 = ${cost_data['baseline']['gpu_cost_usd']:.4f} for {cost_data['baseline']['total_tokens']:,} tokens<br>
                <strong>Candidate:</strong> {cost_data['candidate']['wall_seconds']:.1f} seconds × ${gpu_hourly_rate:.2f}/hour ÷ 3600 = ${cost_data['candidate']['gpu_cost_usd']:.4f} for {cost_data['candidate']['total_tokens']:,} tokens
            </p>
            <p style="font-size: 0.9em; font-style: italic; color: #666;">
                Note: This blended cost amortizes hourly GPU cost across measured throughput. It is not a per-token pricing structure.
            </p>
        """

        throughput_section += cost_section_html
        throughput_section += "</div>"
    else:
        # No GPU rate - throughput only
        throughput_section += """
            <p style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-left: 4px solid #6c757d;">
                <strong>Cost calculation unavailable:</strong> No GPU hourly rate provided. Re-run report generation with <code>--gpu-rate</code> flag to include cost analysis.
            </p>
        </div>
        """

    # Build position summary table
    position_rows = ""
    for pos_name in ['B1', 'C1', 'B2', 'C2', 'B3', 'C3']:
        pos = positions.get(pos_name, {})
        variant = "Baseline" if pos_name.startswith('B') else "Candidate"

        # Get metrics
        measured = 0
        errors = 0
        throughput = 0

        # Get measured requests from request_counts
        if pos.get('conditions'):
            first_cond = pos['conditions'][0]
            measured = first_cond.get('request_counts', {}).get('attempted', 0)

            # Get throughput from best_tested
            if 'pooled_output_tokens_per_second' in pos.get('best_tested', {}):
                throughput = pos['best_tested']['pooled_output_tokens_per_second']

        # Get parameter value for this position
        param_val = "N/A"
        if changed_params:
            first_param = list(changed_params.keys())[0]
            manifest = pos.get('manifest', {})
            flags = manifest.get('engine', {}).get('effective_flags', {})
            param_val = flags.get(first_param, "N/A")

        position_rows += f"""
        <tr>
            <td>{pos_name}</td>
            <td>{variant}</td>
            <td style="text-align: center;">{param_val}</td>
            <td style="text-align: right;">{measured}</td>
            <td style="text-align: right;">{errors}</td>
            <td style="text-align: right;">{throughput:.2f}</td>
        </tr>
        """

    param_header = list(changed_params.keys())[0].replace('_', ' ').title() if changed_params else "Parameter"

    # Full HTML template
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Throttle Benchmark Report</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}

        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            background: #fff;
            padding: 20px;
            max-width: 900px;
            margin: 0 auto;
        }}

        h1, h2, h3 {{
            margin-top: 0;
            margin-bottom: 15px;
            line-height: 1.2;
        }}

        h1 {{
            font-size: 28px;
            border-bottom: 3px solid #000;
            padding-bottom: 10px;
            margin-bottom: 10px;
        }}

        h2 {{
            font-size: 20px;
            margin-top: 30px;
            border-bottom: 2px solid #ddd;
            padding-bottom: 8px;
        }}

        h3 {{
            font-size: 16px;
            margin-top: 20px;
            margin-bottom: 10px;
        }}

        .header {{
            margin-bottom: 30px;
        }}

        .header .subtitle {{
            color: #666;
            font-size: 14px;
            margin-top: 5px;
        }}

        .warning-banner {{
            background: #fff3cd;
            border: 2px solid #ffc107;
            border-radius: 8px;
            padding: 25px;
            margin: 30px 0;
            text-align: center;
        }}

        .warning-banner h2 {{
            color: #856404;
            border: none;
            margin-bottom: 15px;
            font-size: 24px;
        }}

        .warning-banner p {{
            margin: 10px 0;
            font-size: 15px;
        }}

        .section {{
            margin: 40px 0;
        }}

        .recommendation-box {{
            background: #d1ecf1;
            border-left: 5px solid #0c5460;
            padding: 20px;
            margin: 15px 0;
        }}

        .recommendation-text {{
            font-size: 18px;
            margin-bottom: 10px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            font-size: 14px;
        }}

        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}

        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}

        tr:hover {{
            background: #f8f9fa;
        }}

        ul {{
            margin: 15px 0;
            padding-left: 25px;
        }}

        li {{
            margin: 8px 0;
        }}

        code {{
            background: #f4f4f4;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: "Courier New", Courier, monospace;
            font-size: 0.9em;
        }}

        .mono {{
            font-family: "Courier New", Courier, monospace;
            font-size: 0.9em;
        }}

        .footer {{
            margin-top: 60px;
            padding-top: 20px;
            border-top: 2px solid #ddd;
            font-size: 13px;
            color: #666;
            text-align: center;
        }}

        .footer a {{
            color: #007bff;
            text-decoration: none;
        }}

        .footer a:hover {{
            text-decoration: underline;
        }}

        @media print {{
            body {{
                max-width: 100%;
                padding: 10px;
            }}

            .section {{
                page-break-inside: avoid;
            }}
        }}

        @media (max-width: 600px) {{
            body {{
                padding: 10px;
            }}

            h1 {{
                font-size: 22px;
            }}

            table {{
                font-size: 12px;
            }}

            th, td {{
                padding: 8px 4px;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>THROTTLE BENCHMARK REPORT</h1>
        <p class="subtitle">Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>
    </div>

    {warning_banner}

    {recommendation_section}

    {throughput_section}

    <div class="section">
        <h2>3. CONFIGURATION TESTED</h2>

        <h3>Model:</h3>
        <p><code>{model_name}</code> (revision: <span class="mono">{model_revision}...</span>)</p>

        <h3>GPU:</h3>
        <p>{gpu_desc}<br>
        Driver {driver_version}, CUDA {cuda_version}</p>

        <h3>Backend:</h3>
        <p>{engine_name} {engine_version}<br>
        PyTorch {pytorch_version}</p>

        <h3>Parameter Changed:</h3>
        <table>
            <tr>
                <th style="width: 40%;">Parameter</th>
                <th style="text-align: center; width: 30%;">Baseline</th>
                <th style="text-align: center; width: 30%;">Candidate</th>
            </tr>
            {param_summary}
        </table>

        <h3>Workload:</h3>
        <ul>
            <li>{total_requests:,} measured requests ({requests_per_position} per position)</li>
            <li>{total_blocks} blocks total (3 per position)</li>
            <li>Closed-loop concurrency: {concurrency}</li>
            <li>Streaming, temperature 0, max_tokens=128</li>
        </ul>
    </div>

    <div class="section">
        <h2>4. WHY TRUST THIS RESULT</h2>

        <h3>Counterbalanced Protocol (B1/C1/B2/C2/B3/C3):</h3>
        <p>
            Baseline and candidate configurations were tested in alternating order across six positions.
            This controls for time drift - if GPU performance degraded over the session, both configurations
            experience it equally.
        </p>
        <p>
            The statistical method (order-balanced phase contrasts) accounts for position order, ensuring
            the measured difference is due to the configuration change, not when it was measured.
        </p>

        <h3>Decision Gates Passed:</h3>
        <ul>
            <li>✓ All six positions completed successfully</li>
            <li>✓ Zero request errors or cancellations</li>
            <li>✓ Token counts matched between baseline and candidate</li>
            <li>✓ Completion tolerance met (&lt; 5% spread)</li>
            <li>✓ Counterbalanced order verified</li>
            <li>✓ Immutable provenance recorded</li>
        </ul>
    </div>

    <div class="section">
        <h2>5. LIMITS OF THIS RESULT</h2>

        <p>This result is specific to:</p>
        <ul>
            <li>This exact model (<code>{model_name}</code> at revision <span class="mono">{model_revision}</span>)</li>
            <li>This GPU ({gpu_desc})</li>
            <li>This backend version ({engine_name} {engine_version})</li>
            <li>This workload shape (closed-loop concurrency {concurrency}, 128 max tokens)</li>
        </ul>

        <p>It does NOT prove:</p>
        <ul>
            <li>Results will transfer to different models or GPUs</li>
            <li>This is the optimal value for the tested parameter</li>
            <li>Savings will match in production (workload differs)</li>
            <li>Future backend versions will behave identically</li>
        </ul>

        <p>
            This is evidence for a configuration decision, not a universal optimization or projected savings.
        </p>

        <p style="margin-top: 20px; padding: 15px; background: #e7f3ff; border-left: 4px solid #0066cc;">
            <strong>To extend this result:</strong> Run the golden protocol on your own model, GPU, and workload shape.
            Different configurations may yield different outcomes.
        </p>
    </div>

    <div class="section">
        <h2>APPENDIX: SIX-POSITION SUMMARY</h2>

        <table>
            <tr>
                <th>Position</th>
                <th>Variant</th>
                <th style="text-align: center;">{param_header}</th>
                <th style="text-align: right;">Measured</th>
                <th style="text-align: right;">Errors</th>
                <th style="text-align: right;">Output tok/s</th>
            </tr>
            {position_rows}
        </table>
    </div>

    <div class="footer">
        <p>
            <strong>Generated by Throttle</strong><br>
            Run Version: {run_version} | Report Version: {report_version}<br>
            <a href="https://github.com/KushagraKanaujia/throttle">https://github.com/KushagraKanaujia/throttle</a>
        </p>
        <p style="margin-top: 15px;">
            <strong>Questions?</strong> Contact {operator_name} at <a href="mailto:{operator_email}">{operator_email}</a>
        </p>
    </div>
</body>
</html>"""

    return html
