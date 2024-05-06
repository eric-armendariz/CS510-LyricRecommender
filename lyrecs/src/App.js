import './App.css';
import React, { useState } from "react";

// http://127.0.0.1:5000/submit-lyrics

function App() {
  const [lyrics, setLyrics] = useState('');
  const [recommendations, setRecommendations] = useState([]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    try {
      const response = await fetch('http://127.0.0.1:5000/submit-lyrics', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ lyrics })
      });
      const data = await response.json();
      alert('Server says: ' + data.message);
      setRecommendations(data.recommendations);
    } catch (error) {
      console.error('Error submitting lyrics:', error);
      alert('Failed to submit lyrics');
    }
  };

  const handleChange = (event) => {
    setLyrics(event.target.value);
  };

  return (
    <div className="App">
      <header className="App-header">
        <p>Input lyrics here:</p>
        <form onSubmit={handleSubmit}>
          <textarea 
            value={lyrics} 
            onChange={handleChange} 
            rows="4" 
            cols="50" 
            placeholder="Type lyrics here..."
          />
          <button type="submit">Submit</button>
        </form>
        <div>
          <h2>Recommendations:</h2>
          <ul>
            {recommendations.map((recommendation, index) => (
              <li key={index}>{recommendation}</li>
            ))}
          </ul>
        </div>
      </header>
    </div>
  );
}

export default App;
