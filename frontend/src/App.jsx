import { useMemo, useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import styles from './App.module.css'

const API_BASE_URL = 'http://127.0.0.1:8000'

const defaultForm = {
  monthly_salary: 90000,
  age: 30,
  years_of_employment: 6,
  monthly_rent: 18000,
  family_size: 4,
  dependents: 2,
  school_fees: 5000,
  college_fees: 4000,
  travel_expenses: 6000,
  groceries_utilities: 12000,
  other_monthly_expenses: 7000,
  current_emi_amount: 9000,
  credit_score: 745,
  bank_balance: 350000,
  emergency_fund: 120000,
  requested_amount: 1200000,
  requested_tenure: 48,
  annual_interest_rate: 12,
  gender: 'Male',
  marital_status: 'Married',
  education: 'Professional',
  employment_type: 'Private',
  company_type: 'MNC',
  house_type: 'Rented',
  existing_loans: 'Yes',
  emi_scenario: 'Personal Loan EMI',
}

const selectOptions = {
  gender: ['Male', 'Female'],
  marital_status: ['Single', 'Married'],
  education: ['High School', 'Post Graduate', 'Professional'],
  employment_type: ['Private', 'Self-employed'],
  company_type: ['MNC', 'Mid-size', 'Small', 'Startup'],
  house_type: ['Own', 'Rented'],
  existing_loans: ['Yes', 'No'],
  emi_scenario: [
    'Education EMI',
    'Home Appliances EMI',
    'Personal Loan EMI',
    'Vehicle EMI',
  ],
}

const sidebarItems = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'insights', label: 'Insights' },
  { id: 'history', label: 'History' },
]

const currency = new Intl.NumberFormat('en-IN', {
  style: 'currency',
  currency: 'INR',
  maximumFractionDigits: 0,
})

const percent = new Intl.NumberFormat('en-IN', {
  style: 'percent',
  maximumFractionDigits: 1,
})

function toNumber(value) {
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : 0
}

function getRiskLevel(affordabilityRatio) {
  if (affordabilityRatio <= 0.85) {
    return { label: 'Safe', tone: 'safe' }
  }
  if (affordabilityRatio <= 1) {
    return { label: 'Borderline', tone: 'borderline' }
  }
  return { label: 'High Risk', tone: 'risk' }
}

function createTrendPoints(affordableEmi, requiredEmi) {
  const points = []
  const max = Math.max(1, affordableEmi, requiredEmi)
  for (let month = 1; month <= 12; month += 1) {
    const seasonal = Math.sin(month / 2) * 0.06 + 1
    const trend = affordableEmi * seasonal
    const y = 90 - (trend / max) * 70
    const x = (month - 1) * 23 + 6
    points.push(`${x},${Math.max(8, Math.min(92, y)).toFixed(2)}`)
  }
  return points.join(' ')
}

function App() {
  const [activeSidebar, setActiveSidebar] = useState('dashboard')
  const [activeScreen, setActiveScreen] = useState('dashboard')
  const [formData, setFormData] = useState(defaultForm)
  const [result, setResult] = useState(null)
  const [prediction, setPrediction] = useState(null)
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(false)
  const [apiError, setApiError] = useState('')
  const affordableEmi = prediction?.max_monthly_emi_predicted ?? 0
  const requiredEmi = prediction?.formula_monthly_emi_for_requested_loan ?? 0

const affordabilityRatio = requiredEmi > 0
  ? requiredEmi / Math.max(affordableEmi, 1)
  : 0
const burdenRatio = affordableEmi > 0
  ? requiredEmi / affordableEmi
  : 0;

const emiGap = Math.max(requiredEmi - affordableEmi, 0)

//  FIXED risk %
const riskPercent = Math.min(
  Math.max(
    prediction?.risk_percentage ?? ((prediction?.confidence ?? 0) * 100),
    0
  ),
  100
);
  const chartData = [
    { name: 'Affordable EMI', value: affordableEmi },
    { name: 'Required EMI', value: requiredEmi },
]

  const hasChartData = chartData.some((item) => item.value > 0)

  const trendLine = useMemo(
    () => createTrendPoints(affordableEmi || 1, requiredEmi || 1),
    [affordableEmi, requiredEmi],
  )

  const insights = useMemo(() => {
    if (!prediction) {
      return []
    }

    const generated = [
      emiGap > 0
        ? `Your EMI exceeds safe limit by ${currency.format(emiGap)} per month.`
        : 'Your required EMI remains within your affordable monthly capacity.',
      requiredEmi > affordableEmi
        ? 'Try reducing the requested tenure or amount to improve eligibility.'
        : 'Current profile supports the requested loan with manageable risk.',
    ]

    const modelInsights = prediction.explainability?.rule_based_insights ?? []
    return [...generated, ...modelInsights].slice(0, 5)
  }, [prediction, emiGap, requiredEmi, affordableEmi])

  function handleFieldChange(event) {
    const { name, value, type } = event.target
    const nextValue = type === 'number' ? toNumber(value) : value
    setFormData((current) => ({ ...current, [name]: nextValue }))
  }

  async function handleSubmit(event) {
    event.preventDefault()
    setLoading(true)
    setApiError('')

    try {
      const response = await fetch(`${API_BASE_URL}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      })

      if (!response.ok) {
        const errorBody = await response.text()
        console.error("BACKEND ERROR:", errorBody)
        alert(errorBody)
        return
}

      const data = await response.json()
      setPrediction(data)
      setResult({
        affordable_emi: data?.max_monthly_emi_predicted ?? 0,
        required_emi: data?.formula_monthly_emi_for_requested_loan ?? 0,
      })
      setHistory((current) => [{ id: Date.now(), date: new Date(), data }, ...current].slice(0, 6))
      setActiveScreen('dashboard')
      setActiveSidebar('dashboard')
    } catch (error) {
      setApiError(error.message || 'Unable to fetch prediction at the moment.')
    } finally {
      setLoading(false)
    }
  }

  const riskLevel =
    riskPercent < 30
      ? { label: 'Low Risk', color: '#16a34a' }
      : riskPercent < 70
      ? { label: 'Moderate', color: '#f59e0b' }
      : { label: 'High Risk', color: '#dc2626' }

  function renderDashboard() {
  const isEligible = prediction?.eligibility_code === 0;
  const safeRisk = Number(riskPercent ?? 0);

  const riskMeta =
    safeRisk < 30
      ? { label: "Low Risk", color: "#16a34a" }
      : safeRisk < 70
      ? { label: "Moderate Risk", color: "#f59e0b" }
      : { label: "High Risk", color: "#dc2626" };

  const CustomTooltip = ({ active, payload, label }) => {
    if (!active || !payload || payload.length === 0) return null;

    const value = Number(payload[0]?.value ?? 0);

    return (
      <div style={{
        background: '#ffffff',
        border: '1px solid #e2e8f0',
        borderRadius: 8,
        padding: '8px 10px',
        boxShadow: '0 6px 24px rgba(15, 23, 42, 0.08)',
      }}>
        <p style={{ margin: 0, fontSize: 12 }}>{label}</p>
        <p style={{ margin: '4px 0 0', fontWeight: 700 }}>
          {currency.format(value)}
        </p>
      </div>
    );
  };

  return (
    <>
      {/* METRICS */}
      <section className={styles.metricsGrid}>

        <article className={styles.metricCard}>
          <h3>Maximum Affordable EMI</h3>
          <p className={styles.metricValue}>
            {affordableEmi === 0
              ? "Not Affordable"
              : currency.format(affordableEmi)}
          </p>
        </article>

        <article className={styles.metricCard}>
          <h3>Required EMI</h3>
          <p className={styles.metricValue}>{currency.format(requiredEmi)}</p>
        </article>

        <article className={styles.metricCard}>
          <h3>Risk %</h3>
          <p className={styles.metricValue}>
            {prediction ? `${riskPercent.toFixed(1)}%` : '--'}
          </p>
        </article>
      </section>

      {/* COMPARISON */}
      <section className={styles.sectionCard}>
        <h2>Financial Comparison</h2>

        {hasChartData ? (
          <div style={{ width: '100%', height: 280 }}>
            <ResponsiveContainer>
              <BarChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" />
                <YAxis tickFormatter={(v) => currency.format(v)} />
                <Tooltip content={<CustomTooltip />} />

                <Bar dataKey="value">
                  {chartData.map((entry) => (
                    <Cell
                      key={entry.name}
                      fill={entry.name === 'Affordable EMI' ? '#16a34a' : '#2563eb'}
                    />
                  ))}
                  <LabelList
                    dataKey="value"
                    position="top"
                    formatter={(v) => currency.format(v)}
                  />
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <p>Run prediction to see chart</p>
        )}

        {/*  SMART MESSAGE */}
        <p style={{
          marginTop: 10,
          color: requiredEmi > affordableEmi ? '#dc2626' : '#16a34a',
          fontWeight: 600
        }}>
          {requiredEmi > affordableEmi
            ? ` EMI exceeds affordability by ${currency.format(emiGap)}`
            : ` EMI is within safe limit`}
        </p>

        {/*  FINAL DECISION */}
        <p style={{ fontWeight: 600 }}>
          {prediction
            ? isEligible
              ? " You can safely take this loan"
              : " High financial risk — reconsider loan terms"
            : "Run prediction to see decision"}
        </p>
      </section>

      {/* RISK + TREND */}
      <section className={styles.twoColumn}>
        <article className={styles.sectionCard}>
          <h2>Risk Analysis</h2>

          <div>
       <p>
         EMI Burden Ratio: {burdenRatio.toFixed(2)}
       </p>
            <p>Status: {riskLevel.label}</p>
          </div>

          {/*  RISK METER */}
          <div style={{ marginTop: 10 }}>
            <div style={{
              height: 10,
              background: '#e2e8f0',
              borderRadius: 5
            }}>
              <div style={{
                width: `${riskPercent.toFixed(1)}%`,
                height: '100%',
                borderRadius: 5,
                background:
                  riskPercent < 30 ? '#16a34a' :
                  riskPercent < 70 ? '#f59e0b' :
                  '#dc2626'
              }} />
            </div>
          </div>

          
          <p style={{ marginTop: 8, color: riskMeta.color, fontWeight: "bold" }}>
            {safeRisk.toFixed(1)}% ({riskMeta.label})
          </p>

          {/*  DECISION FACTORS */}
          {prediction?.explainability?.decision_factors?.length > 0 && (
            <>
              <h4 style={{ marginTop: 10 }}>Recommendations</h4>
              <ul>
                {requiredEmi > affordableEmi && (
                  <li>Reduce loan amount or increase tenure</li>
                )}



                {prediction?.confidence < 0.5 && (
                  <li>Maintain a higher savings buffer before applying</li>
                )}

                {affordableEmi === 0 && (
                  <li>Current financials do not support additional EMI — avoid loan</li>
                )}
              </ul>
            </>
          )}

          {/*  WHY THIS DECISION */}
          <h4>Recommended Actions</h4>
          <ul>
            {prediction?.confidence > 0.6 && (
              <li>High risk — consider reducing loan amount</li>
           )}

           {prediction?.max_monthly_emi_predicted <
             prediction?.formula_monthly_emi_for_requested_loan && (
             <li>Requested EMI exceeds affordability — increase tenure or reduce loan</li>
            )}

            {prediction?.confidence > 0.5 && (
              <li>Financial stress is moderate/high — proceed cautiously</li>
            )}

            <li>Maintain higher savings to improve eligibility</li>
          </ul>
        </article>

        <article className={styles.sectionCard}>
          <h2>EMI Trend</h2>
          <span>Simulated</span>

          <svg viewBox="0 0 270 100">
            <polyline points={trendLine} />
          </svg>
        </article>
      </section>
    </>
  );
}
    
       
                 
     

  function renderInsights() {
    return (
      <section className={styles.sectionCard}>
        <div className={styles.sectionHeader}>
          <h2>Insights</h2>
          <span className={styles.inlinePill}>Actionable</span>
        </div>
        <ul className={styles.insightList}>
          {insights.length > 0 ? (
            insights.map((item) => <li key={item}>{item}</li>)
          ) : (
            <li>Run a prediction from the input page to view smart suggestions.</li>
          )}
        </ul>
      </section>
    )
  }

  function renderHistory() {
    return (
      <section className={styles.sectionCard}>
        <div className={styles.sectionHeader}>
          <h2>History</h2>
          <span className={styles.inlinePill}>Last 6 Predictions</span>
        </div>
        <div className={styles.historyTable}>
          <div className={styles.historyHead}>
            <span>Timestamp</span>
            <span>Status</span>
            <span>Affordable EMI</span>
            <span>Required EMI</span>
          </div>
          {history.length === 0 ? (
            <p className={styles.emptyState}>No predictions yet.</p>
          ) : (
            history.map((item) => (
              <div key={item.id} className={styles.historyRow}>
                <span>{new Date(item.date).toLocaleString('en-IN')}</span>
                <span>{item.data.eligibility_label}</span>
                <span>{currency.format(item.data.max_monthly_emi_predicted)}</span>
                <span>{currency.format(item.data.formula_monthly_emi_for_requested_loan)}</span>
              </div>
            ))
          )}
        </div>
      </section>
    )
  }

  function renderInputPage() {
    return (
      <section className={styles.formPage}>
        <div className={styles.sectionHeader}>
          <h2>Loan Assessment Input</h2>
          <span className={styles.inlinePill}>Structured Form</span>
        </div>

        <form className={styles.formGrid} onSubmit={handleSubmit}>
          <fieldset>
            <legend>Income</legend>
            <label>
              Monthly Salary
              <input type="number" name="monthly_salary" value={formData.monthly_salary} onChange={handleFieldChange} min="1" required />
            </label>
            <label>
              Bank Balance
              <input type="number" name="bank_balance" value={formData.bank_balance} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              Emergency Fund
              <input type="number" name="emergency_fund" value={formData.emergency_fund} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              Credit Score
              <input type="number" name="credit_score" value={formData.credit_score} onChange={handleFieldChange} min="300" max="900" required />
            </label>
          </fieldset>

          <fieldset>
            <legend>Expenses</legend>
            <label>
              Monthly Rent
              <input type="number" name="monthly_rent" value={formData.monthly_rent} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              Current EMI
              <input type="number" name="current_emi_amount" value={formData.current_emi_amount} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              Groceries & Utilities
              <input type="number" name="groceries_utilities" value={formData.groceries_utilities} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              Travel Expenses
              <input type="number" name="travel_expenses" value={formData.travel_expenses} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              School Fees
              <input type="number" name="school_fees" value={formData.school_fees} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              College Fees
              <input type="number" name="college_fees" value={formData.college_fees} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              Other Expenses
              <input type="number" name="other_monthly_expenses" value={formData.other_monthly_expenses} onChange={handleFieldChange} min="0" required />
            </label>
          </fieldset>

          <fieldset>
            <legend>Loan Details</legend>
            <label>
              Requested Amount
              <input type="number" name="requested_amount" value={formData.requested_amount} onChange={handleFieldChange} min="1" required />
            </label>
            <label>
              Requested Tenure (Months)
              <input type="number" name="requested_tenure" value={formData.requested_tenure} onChange={handleFieldChange} min="1" required />
            </label>
            <label>
              Annual Interest Rate
              <input type="number" name="annual_interest_rate" value={formData.annual_interest_rate} onChange={handleFieldChange} min="1" required />
            </label>
            <label>
              EMI Scenario
              <select name="emi_scenario" value={formData.emi_scenario} onChange={handleFieldChange}>
                {selectOptions.emi_scenario.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </label>
          </fieldset>

          <fieldset>
            <legend>Personal Info</legend>
            <label>
              Years of Employment
              <input type="number" name="years_of_employment" value={formData.years_of_employment} onChange={handleFieldChange} min="0" required />
            </label>
            <label>
              Family Size
              <input type="number" name="family_size" value={formData.family_size} onChange={handleFieldChange} min="1" required />
            </label>
            <label>
              Dependents
              <input type="number" name="dependents" value={formData.dependents} onChange={handleFieldChange} min="0" required />
            </label>

            {Object.entries(selectOptions)
              .filter(([key]) => key !== 'emi_scenario')
              .map(([key, options]) => (
                <label key={key}>
                  {key.replace(/_/g, ' ')}
                  <select name={key} value={formData[key]} onChange={handleFieldChange}>
                    {options.map((option) => (
                      <option key={option} value={option}>
                        {option}
                      </option>
                    ))}
                  </select>
                </label>
              ))}
          </fieldset>

          <div className={styles.formActions}>
            <button type="button" onClick={() => setFormData(defaultForm)} className={styles.secondaryBtn}>
              Reset
            </button>
            <button type="submit" className={styles.primaryBtn} disabled={loading}>
              {loading ? 'Predicting...' : 'Run EMI Prediction'}
            </button>
          </div>
        </form>

        {apiError && <p className={styles.apiError}>{apiError}</p>}
      </section>
    )
  }

  return (
    <div className={styles.appShell}>
      <aside className={styles.sidebar}>
        <h1>EMI Predict AI</h1>
        <nav>
          {sidebarItems.map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setActiveSidebar(item.id)
                setActiveScreen(item.id)
              }}
              className={activeSidebar === item.id ? styles.activeNavItem : styles.navItem}
            >
              {item.label}
            </button>
          ))}
        </nav>
        <button type="button" onClick={() => setActiveScreen('input')} className={styles.primaryBtn}>
          New Prediction
        </button>
      </aside>

      <main className={styles.mainContent}>
        <header className={styles.topbar}>
          <div>
            <h2>Fintech Loan Intelligence Dashboard</h2>
            <p>Real-time eligibility and EMI risk monitoring</p>
          </div>
          <div className={styles.userIcon} aria-label="User profile">
            RU
          </div>
        </header>

        {activeScreen === 'input' && renderInputPage()}
        {activeScreen === 'dashboard' && renderDashboard()}
        {activeScreen === 'insights' && renderInsights()}
        {activeScreen === 'history' && renderHistory()}
      </main>
    </div>
  )
}

export default App
