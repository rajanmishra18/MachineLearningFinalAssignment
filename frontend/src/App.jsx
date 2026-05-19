import { useState } from "react";
import axios from "axios";
import "./App.css";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  LineChart,
  Line,
  Cell,
  Legend,
  ResponsiveContainer
} from "recharts";

function App() {
  const [model, setModel] = useState("logistic_regression");
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    setLoading(true);
    try {
      const response = await axios.post(
        `${import.meta.env.VITE_API_URL}/predict`,
        null,
        { params: { model_name: model } }
      );
      setData(response.data);
    } catch (error) {
      console.error(error);
    }
    setLoading(false);
  };

  return (
    <div className="container">

      <div className="header">
        ML Model Dashboard
      </div>

      {/* Controls */}
      <div className="card">
        <div className="controls">
          <select onChange={(e) => setModel(e.target.value)}>
            <option value="logistic_regression">Logistic Regression</option>
            <option value="linear_regression">Linear Regression</option>
            <option value="decision_tree">Decision Tree</option>
            <option value="knn">KNN</option>
            <option value="svm">SVM</option>
          </select>

          <button onClick={handlePredict}>
            {loading ? "Running..." : "Run Model"}
          </button>
        </div>
      </div>

      {/* ================= CLASSIFICATION ================= */}
      {data && data.type === "classification" && (
        <>
          {/* Metrics */}
          <div className="card">
            <h3>Classification Performance</h3>

            <div style={{ display: "flex", flexWrap: "wrap", gap: "10px" }}>
              {data.metrics?.accuracy && (
                <div><strong>Accuracy:</strong> {data.metrics.accuracy.toFixed(2)}</div>
              )}
              {data.metrics?.precision && (
                <div><strong>Precision:</strong> {data.metrics.precision.toFixed(2)}</div>
              )}
              {data.metrics?.recall && (
                <div><strong>Recall:</strong> {data.metrics.recall.toFixed(2)}</div>
              )}
            </div>
          </div>

          {/* GRID START */}
          <div className="grid">

            {/* Confusion Matrix */}
            {data.metrics?.confusion_matrix && (
              <div className="card">
                <h3>Confusion Matrix</h3>

                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={[
                      { name: "TP", value: data.metrics.confusion_matrix.TP },
                      { name: "TN", value: data.metrics.confusion_matrix.TN },
                      { name: "FP", value: data.metrics.confusion_matrix.FP },
                      { name: "FN", value: data.metrics.confusion_matrix.FN }
                    ]}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value">
                      <Cell fill="#4CAF50" />
                      <Cell fill="#2E7D32" />
                      <Cell fill="#FF9800" />
                      <Cell fill="#F44336" />
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Prediction Distribution */}
            {data.predictions && (
              <div className="card">
                <h3>Prediction Distribution</h3>

                <ResponsiveContainer width="100%" height={300}>
                  <BarChart
                    data={[
                      {
                        name: "Class 0",
                        value: data.predictions.filter(p => p === 0).length
                      },
                      {
                        name: "Class 1",
                        value: data.predictions.filter(p => p === 1).length
                      }
                    ]}
                  >
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="value" fill="#2196F3" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            {/* Accuracy */}
            {data.metrics?.accuracy && (
              <div className="card">
                <h3>Accuracy</h3>

                <ResponsiveContainer width="100%" height={200}>
                  <BarChart
                    data={[
                      { name: data.model, accuracy: data.metrics.accuracy }
                    ]}
                  >
                    <XAxis dataKey="name" />
                    <YAxis domain={[0, 1]} />
                    <Tooltip />
                    <Bar dataKey="accuracy" fill="#4CAF50" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

          </div>
        </>
      )}

      {/* ================= REGRESSION ================= */}
      {data && data.type === "regression" && (
        <>
          <div className="card">
            <h3>Regression Performance</h3>
            <p><strong>MSE:</strong> {data.metrics?.mse.toFixed(4)}</p>
          </div>

          <div className="grid">
            <div className="card">
              <h3>Actual vs Predicted</h3>

              <ResponsiveContainer width="100%" height={300}>
                <LineChart
                  data={data.actual.map((val, i) => ({
                    index: i,
                    actual: val,
                    predicted: data.predictions[i]
                  }))}
                >
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="index" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Line type="monotone" dataKey="actual" stroke="#4CAF50" />
                  <Line type="monotone" dataKey="predicted" stroke="#2196F3" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}

    </div>
  );
}

export default App;