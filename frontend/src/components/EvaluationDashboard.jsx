import { useCallback, useEffect, useState } from "react";

import { api } from "../services/api.js";

function formatMetric(value, digits = 3) {
  return typeof value === "number" && Number.isFinite(value)
    ? value.toFixed(digits)
    : "Unavailable";
}

function formatLatency(value) {
  return typeof value === "number" && Number.isFinite(value)
    ? `${value.toFixed(1)} ms`
    : "Unavailable";
}

function metricRows(run) {
  if (!run) return [];
  const retrieval = run.aggregates?.retrieval || {};
  const quality = run.aggregates?.answer_quality || {};
  const latency = run.aggregates?.latency || {};
  return [
    ...Object.entries(retrieval).map(([name, value]) => ({
      name: `Retrieval · ${name}`,
      value: value.mean,
      n: value.n,
    })),
    ...Object.entries(quality).map(([name, value]) => ({
      name: `Answer · ${name}`,
      value: value.mean,
      n: value.n,
    })),
    ...["retrieval", "generation", "total"].map((stage) => ({
      name: `Latency · ${stage} median`,
      value: latency[stage]?.median_ms,
      unit: "ms",
      n: latency[stage]?.n,
    })),
    ...["retrieval", "generation", "total"].map((stage) => ({
      name: `Latency · ${stage} p95`,
      value: latency[stage]?.p95_ms,
      unit: "ms",
      n: latency[stage]?.n,
    })),
  ];
}

function RunMetrics({ run }) {
  if (!run) return <p className="eval-empty">Select an evaluation run to view its metrics.</p>;
  const rows = metricRows(run);
  return (
    <div className="eval-table-wrap">
      <table className="eval-table">
        <thead><tr><th>Metric</th><th>Value</th><th>Scored cases</th></tr></thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.name}>
              <td>{row.name}</td>
              <td>{row.unit === "ms" ? formatLatency(row.value) : formatMetric(row.value)}</td>
              <td>{Number.isInteger(row.n) ? row.n : "—"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function FailureList({ run }) {
  const failures = run?.failures || [];
  if (!run) return null;
  if (!failures.length) return <p className="eval-empty">No failure or citation-mismatch cases were recorded.</p>;
  return (
    <div className="eval-failures">
      {failures.map((failure) => (
        <article className="eval-failure" key={failure.case_id}>
          <div>
            <strong>{failure.case_id}</strong>
            <span>{failure.failure_kinds.join(", ")}</span>
          </div>
          <p>
            Expected pages: {failure.expected.pages.join(", ") || "none"} ·
            Actual cited pages: {failure.actual.pages.join(", ") || "none"} ·
            Status: {failure.actual.status || "error"}
          </p>
        </article>
      ))}
    </div>
  );
}

function RunLabel({ metadata }) {
  if (!metadata) return "Select a run";
  return `${metadata.dataset_version} · ${metadata.generated_at}`;
}

export default function EvaluationDashboard() {
  const [runs, setRuns] = useState([]);
  const [primaryId, setPrimaryId] = useState("");
  const [comparisonId, setComparisonId] = useState("");
  const [primary, setPrimary] = useState(null);
  const [comparison, setComparison] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const response = await api.evaluationRuns();
      setRuns(response.runs);
      setPrimaryId((current) =>
        response.runs.some((run) => run.metadata.run_id === current)
          ? current
          : response.runs[0]?.metadata.run_id || "");
      setComparisonId((current) =>
        response.runs.some((run) => run.metadata.run_id === current)
          ? current
          : response.runs[1]?.metadata.run_id || "");
    } catch (requestError) {
      setError(requestError.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  useEffect(() => {
    let active = true;
    if (!primaryId) {
      setPrimary(null);
      return () => { active = false; };
    }
    api.evaluationRun(primaryId)
      .then((response) => { if (active) setPrimary(response.run); })
      .catch((requestError) => { if (active) setError(requestError.message); });
    return () => { active = false; };
  }, [primaryId]);

  useEffect(() => {
    let active = true;
    if (!comparisonId || comparisonId === primaryId) {
      setComparison(null);
      return () => { active = false; };
    }
    api.evaluationRun(comparisonId)
      .then((response) => { if (active) setComparison(response.run); })
      .catch((requestError) => { if (active) setError(requestError.message); });
    return () => { active = false; };
  }, [comparisonId, primaryId]);

  const primaryRows = metricRows(primary);
  const comparisonByName = new Map(metricRows(comparison).map((row) => [row.name, row]));
  const runOptions = runs.map((run) => ({
    id: run.metadata.run_id,
    label: <RunLabel metadata={run.metadata} />,
  }));

  return (
    <section className="panel evaluation-dashboard" aria-labelledby="evaluation-title">
      <div className="eval-heading">
        <div>
          <p className="eyebrow">RETRIEVAL · ANSWERS · PERFORMANCE</p>
          <h2 id="evaluation-title">Evaluation runs</h2>
          <p>Review local RAG evaluation reports. Run evaluations with the CLI; this dashboard is read-only.</p>
        </div>
        <button className="secondary-button" type="button" onClick={refresh} disabled={loading}>
          {loading ? "Refreshing…" : "Refresh reports"}
        </button>
      </div>

      {error && <div className="notice notice-error" role="alert">{error}</div>}
      {!loading && !runs.length && !error && (
        <div className="eval-empty-state">
          <span aria-hidden="true">▤</span>
          <strong>No evaluation reports yet</strong>
          <p>Run <code>python -m evaluation.cli</code> from the project root to create a report.</p>
        </div>
      )}

      {runs.length > 0 && (
        <>
          <div className="eval-run-controls">
            <label>
              Evaluation run
              <select value={primaryId} onChange={(event) => setPrimaryId(event.target.value)}>
                {runOptions.map((option) => <option key={option.id} value={option.id}>{option.label}</option>)}
              </select>
            </label>
            <label>
              Compare with
              <select value={comparisonId} onChange={(event) => setComparisonId(event.target.value)}>
                <option value="">No comparison</option>
                {runOptions.filter((option) => option.id !== primaryId).map((option) => (
                  <option key={option.id} value={option.id}>{option.label}</option>
                ))}
              </select>
            </label>
          </div>

          {primary && (
            <div className="eval-run-meta">
              <span><strong>Dataset:</strong> {primary.metadata.dataset_version}</span>
              <span><strong>Cases:</strong> {primary.metadata.case_count}</span>
              <span><strong>Embedding:</strong> {primary.metadata.embedding_model}</span>
              <span><strong>Generation:</strong> {primary.metadata.generation_model}</span>
              <span><strong>Cost:</strong> Not applicable (local Ollama)</span>
              {primary.metadata.local_judge_is_subjective && <span className="eval-subjective">Local-model scores are subjective</span>}
            </div>
          )}

          <h3 className="eval-section-heading">Metric summary</h3>
          {comparison ? (
            <div className="eval-table-wrap">
              <table className="eval-table">
                <thead><tr><th>Metric</th><th>Selected run</th><th>Comparison run</th></tr></thead>
                <tbody>
                  {primaryRows.map((row) => {
                    const other = comparisonByName.get(row.name);
                    return (
                      <tr key={row.name}>
                        <td>{row.name}</td>
                        <td>{row.unit === "ms" ? formatLatency(row.value) : formatMetric(row.value)}</td>
                        <td>{!other ? "Unavailable" : row.unit === "ms" ? formatLatency(other.value) : formatMetric(other.value)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : <RunMetrics run={primary} />}

          <h3 className="eval-section-heading">Failure examples and citation mismatches</h3>
          <FailureList run={primary} />
          {comparison && (
            <>
              <h3 className="eval-section-heading">Comparison run failures</h3>
              <FailureList run={comparison} />
            </>
          )}
          <p className="eval-disclaimer">
            Retrieval is scored at source-page level unless verified chunk labels exist. Answer text and source contents are omitted from reports.
            Lexical answer metrics are proxies, not semantic ground truth. p95 latency is unavailable for fewer than 20 samples.
          </p>
        </>
      )}
    </section>
  );
}
