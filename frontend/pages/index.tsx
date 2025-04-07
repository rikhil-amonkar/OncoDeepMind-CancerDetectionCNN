import { useState } from 'react';

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

    const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    }

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
        setPredict(data);
    }

    return (
        <div className='flex flex-col items-center justify-center min-h-screen py-2'>
            <h1 className='text-4xl font-bold mb-4'>Predict Drug Response</h1>
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

      {prediction && (
        <div className="mt-6 p-4 border rounded bg-gray-100">
          <p><strong>AUC:</strong> {prediction.auc.toFixed(5)}</p>
          <p><strong>Effectiveness:</strong> {prediction.percent_effectiveness}%</p>
        </div>
      )}
    </div>
  );
}


