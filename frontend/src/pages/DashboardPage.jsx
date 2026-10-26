import { useEffect, useState } from 'react';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { AlertTriangle, Database, Network, ShieldAlert } from 'lucide-react';
import EmptyState from '../components/EmptyState.jsx';
import ErrorState from '../components/ErrorState.jsx';
import LoadingState from '../components/LoadingState.jsx';
import PageHeader from '../components/PageHeader.jsx';
import SeverityBadge from '../components/SeverityBadge.jsx';
import StatCard from '../components/StatCard.jsx';
import { getDashboardSummary } from '../api/dashboardApi.js';

const severityLevels = ['critical', 'high', 'medium', 'low', 'info'];
const predictionTypes = ['known_attack', 'benign', 'anomaly', 'uncertain'];
const countFormatter = new Intl.NumberFormat();

function formatCount(value) {
  return typeof value === 'number' && Number.isFinite(value)
    ? countFormatter.format(value)
    : '—';
}

function formatRiskScore(value) {
  return typeof value === 'number' && Number.isFinite(value)
    ? new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value)
    : 'Unavailable';
}

function formatTimestamp(value) {
  if (typeof value !== 'string' || !value) return 'Unavailable';

  const timestamp = new Date(value);
  if (Number.isNaN(timestamp.getTime())) return 'Unavailable';

  return new Intl.DateTimeFormat(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  }).format(timestamp);
}

function formatPredictionType(value) {
  return value.replaceAll('_', ' ');
}

function RiskTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;

  const values = Object.fromEntries(
    payload.map((entry) => [entry.dataKey, entry.value]),
  );

  return (
    <div className="chart-tooltip">
      <strong>{formatTimestamp(label)}</strong>
      <span>Observations: {formatCount(values.count)}</span>
      <span>
        Mean risk score:{' '}
        {formatRiskScore(values.mean_risk_score)}
      </span>
    </div>
  );
}

export default function DashboardPage() {
  const [summary, setSummary] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [requestVersion, setRequestVersion] = useState(0);

  useEffect(() => {
    let isCurrentRequest = true;

    async function loadSummary() {
      setLoading(true);
      setError('');

      try {
        const result = await getDashboardSummary();
        if (isCurrentRequest) setSummary(result);
      } catch (requestError) {
        if (isCurrentRequest) {
          setError(
            requestError?.userMessage
              || 'The dashboard summary could not be loaded. Please try again.',
          );
        }
      } finally {
        if (isCurrentRequest) setLoading(false);
      }
    }

    loadSummary();

    return () => {
      isCurrentRequest = false;
    };
  }, [requestVersion]);

  if (loading) {
    return (
      <div className="dashboard-page dashboard-loading">
        <LoadingState label="Loading dashboard summary…" />
      </div>
    );
  }

  if (error) {
    return (
      <div className="dashboard-page">
        <PageHeader
          eyebrow="SECURITY OVERVIEW"
          title="Network overview"
          description="Backend-provided network security activity."
        />
        <ErrorState
          title="Dashboard summary unavailable"
          message={error}
          onRetry={() => setRequestVersion((version) => version + 1)}
        />
      </div>
    );
  }

  const totals = summary.totals;
  const riskSeries = Array.isArray(summary.risk_series)
    ? summary.risk_series
    : [];
  const hasNoActivity = [
    totals.datasets,
    totals.observations,
    totals.alerts,
    totals.open_alerts,
  ].every((value) => value === 0) && riskSeries.length === 0;
  const unavailableRiskScores = riskSeries.filter(
    (point) => point.mean_risk_score == null,
  ).length;

  return (
    <div className="dashboard-page">
      <PageHeader
        eyebrow="SECURITY OVERVIEW"
        title="Network overview"
        description="Backend-provided network security activity."
      >
        <div className="environment-pill">
          <span className="status-dot status-muted" />
          SUMMARY DATA
        </div>
      </PageHeader>

      <div className="stat-grid" role="group" aria-label="Dashboard totals">
        <StatCard
          label="Observed traffic"
          value={formatCount(totals.observations)}
          note="Total observations"
          icon={Network}
        />
        <StatCard
          label="Alerts"
          value={formatCount(totals.alerts)}
          note="Total alerts"
          icon={ShieldAlert}
        />
        <StatCard
          label="Datasets"
          value={formatCount(totals.datasets)}
          note="Datasets included"
          icon={Database}
        />
        <StatCard
          label="Open alerts"
          value={formatCount(totals.open_alerts)}
          note="Alerts currently open"
          icon={AlertTriangle}
        />
      </div>

      {hasNoActivity && (
        <div className="dashboard-empty-notice">
          <EmptyState
            title="No activity for this summary"
            message="The backend returned zero totals and no risk-series data."
          />
        </div>
      )}

      <section className="dashboard-section" aria-labelledby="risk-heading">
        <div className="section-heading">
          <div>
            <h2 id="risk-heading">Risk trend</h2>
            <p>Observation counts and mean risk score by time bucket</p>
          </div>
          <span className="section-state">
            {formatTimestamp(summary.range?.from)} – {formatTimestamp(summary.range?.to)}
          </span>
        </div>
        {riskSeries.length > 0 ? (
          <div className="risk-chart" role="img" aria-label="Risk trend chart">
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={riskSeries} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
                <CartesianGrid stroke="#1d2a3d" strokeDasharray="3 3" />
                <XAxis
                  dataKey="bucket_start"
                  tickFormatter={formatTimestamp}
                  stroke="#8290a8"
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  yAxisId="count"
                  allowDecimals={false}
                  stroke="#8290a8"
                  tick={{ fontSize: 10 }}
                />
                <YAxis
                  yAxisId="risk"
                  orientation="right"
                  stroke="#8290a8"
                  tick={{ fontSize: 10 }}
                />
                <Tooltip content={<RiskTooltip />} />
                <Legend />
                <Line
                  yAxisId="count"
                  type="monotone"
                  dataKey="count"
                  name="Observations"
                  stroke="#53e0d0"
                  dot={false}
                />
                <Line
                  yAxisId="risk"
                  type="monotone"
                  dataKey="mean_risk_score"
                  name="Mean risk score"
                  stroke="#70a5ff"
                  dot={false}
                  connectNulls={false}
                />
              </LineChart>
            </ResponsiveContainer>
            {unavailableRiskScores > 0 && (
              <p className="chart-note">
                Mean risk score is unavailable for {unavailableRiskScores}{' '}
                {unavailableRiskScores === 1 ? 'time bucket' : 'time buckets'}.
              </p>
            )}
          </div>
        ) : (
          <EmptyState
            title="No risk-series data"
            message="The backend returned no time buckets for the risk trend."
          />
        )}
      </section>

      <div className="dashboard-bottom-grid">
        <section className="dashboard-section compact-section" aria-labelledby="severity-heading">
          <div className="section-heading">
            <div>
              <h2 id="severity-heading">Alerts by severity</h2>
              <p>Severity counts included in this summary</p>
            </div>
          </div>
          <dl className="dashboard-count-list">
            {severityLevels.map((severity) => (
              <div className="dashboard-count-row" key={severity}>
                <dt><SeverityBadge severity={severity} /></dt>
                <dd>{formatCount(summary.severity_counts[severity])}</dd>
              </div>
            ))}
          </dl>
        </section>

        <section className="dashboard-section compact-section" aria-labelledby="prediction-heading">
          <div className="section-heading">
            <div>
              <h2 id="prediction-heading">Predictions by type</h2>
              <p>Prediction counts included in this summary</p>
            </div>
          </div>
          <dl className="dashboard-count-list">
            {predictionTypes.map((type) => (
              <div className="dashboard-count-row" key={type}>
                <dt>{formatPredictionType(type)}</dt>
                <dd>{formatCount(summary.prediction_type_counts[type])}</dd>
              </div>
            ))}
          </dl>
        </section>
      </div>

      <footer className="page-footer">
        <span>ADAPTIVE AI NIDS</span>
        <span>Summary generated {formatTimestamp(summary.generated_at)}</span>
      </footer>
    </div>
  );
}
