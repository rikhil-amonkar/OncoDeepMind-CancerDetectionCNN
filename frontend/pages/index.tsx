import { useState } from 'react';
import AsyncSelect from 'react-select/async';
import Select from 'react-select';

export default function Home() {
    const [form, setForm] = useState({
        CELL_LINE_NAME: '',
        TCGA_DESC: '',
        CNA: '',
        Gene_Expression: '',
        Methylation: '',
        MSI_Status: '',
        Growth_Properties: '',
        Screen_Medium: '',
        Cancer_Type: '',
        TARGET: '',
        TARGET_PATHWAY: ''
    });


    const [prediction, setPredict] = useState<null | {auc: number, percent_effectiveness: number}>(null);

    const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    }

    // Assign some dropdown options for each input
    const options = {
        CELL_LINE_NAME: ['Cell Line A', 'Cell Line B', 'Cell Line C'],
        TCGA_DESC: ['Description A', 'Description B', 'Description C'],
        CNA: ['Low', 'Medium', 'High'],
        Gene_Expression: ['Low', 'Normal', 'High'],
        Methylation: ['Low', 'Moderate', 'High'],
        MSI_Status: ['Stable', 'Instable'],
        Growth_Properties: ['Slow', 'Normal', 'Fast'],
        Screen_Medium: ['Medium A', 'Medium B', 'Medium C'],
        Cancer_Type: ['Lung', 'Breast', 'Colon'],
        TARGET: ['Target X', 'Target Y', 'Target Z'],
        TARGET_PATHWAY: ['Pathway A', 'Pathway B', 'Pathway C']
    };

    const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
        e.preventDefault();
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(form)
        });
        const data = await response.json();
        setPredict({
            auc: data['Predicted AUC'],
            percent_effectiveness: data['Percent Effectiveness']
        });
    }

    return (
        <div className="p-6">
          <h1 className="text-2xl font-bold mb-4">Drug Response Predictor</h1>
          <form onSubmit={handleSubmit} className="grid grid-cols-2 gap-4">
            {Object.keys(form).map((key) => (
              <input
                key={key}
                name={key}
                value={form[key as keyof typeof form]}
                onChange={handleChange}
                className="border p-2"
                placeholder={key}
              />
            ))}
            <button className="col-span-2 bg-blue-600 text-white p-2 rounded" type="submit">
              Predict AUC
            </button>
          </form>
    
        {prediction ? (
            <div className="mt-6 p-4 border rounded bg-gray-100">
                <p><strong>AUC:</strong> {prediction.auc ? prediction.auc.toFixed(5) : 'N/A'}</p>
                <p><strong>Effectiveness:</strong> {prediction.percent_effectiveness ? prediction.percent_effectiveness.toFixed(2) : 'N/A'}%</p>
            </div>
        ) : (
            <p>Waiting for prediction...</p>
        )}
        </div>
      );
    };


