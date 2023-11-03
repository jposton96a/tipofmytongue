import React, { useState } from 'react';
import logo from './logo.svg';
// import './App.css';
import styles from './Home.module.css'

async function getWords(operators: WordOperation[]): Promise<WordPrediction[]> {
  const res = await fetch('/operations', {
    method: 'POST',
    headers: {
      'Content-type': 'application/json',
    },
    body: JSON.stringify(operators),
  })
  
  if(!res.ok){
    throw new Error(`Response failed with code ${res.status} - ${res.statusText}`);
  }
  
  const parsedResp = await res.json() as unknown as WordOperation[];
  console.log("Response: ", parsedResp)
  if(parsedResp.length >= 1) {
    console.log("Response results:", parsedResp[parsedResp.length - 1].results)
   
    // Return the results directly or an empty array 
    return parsedResp[parsedResp.length - 1].results || [] as WordPrediction[];
  }

  return [];
}

type WordAction = "start" | "more_like" | "less_like"

interface WordPrediction { word: string; dist: number };

type WordOperation = {
  id?: string,
  function: WordAction,
  description: string,
  results?: WordPrediction[],
  selected_words?: string[]
}

const clone = (obj: any) => JSON.parse(JSON.stringify(obj));

const initialAppState = {
  ops: [{ description: '', function: 'start' }] as WordOperation[],
  predictions: []
}

function App() {
  const [operators, setOperators] = useState<WordOperation[]>(clone(initialAppState.ops));
  const [predictions, setPredictions] = useState<WordPrediction[]>(clone(initialAppState.predictions));

  const handleAddOperator = () => {
    setOperators([...operators, { description: '', function: 'more_like' }]);
  };

  const handleOperatorChange = (index: number, field: keyof WordOperation, value: any) => {
    const updatedOperators = [...operators];
    updatedOperators[index][field] = value;
    setOperators(updatedOperators);
    // calculateWords(updatedOperators);
  };

  const handleRemoveOperator = (index: number) => {
    const updatedOperators = [...operators];
    updatedOperators.splice(index, 1);
    setOperators(updatedOperators);
    // calculateWords(updatedOperators);
  };

  const calculateWords = (operators: WordOperation[]) => {
    operators = operators.filter(operator => operator.description !== "" && operator.description !== null);
    
    getWords(operators).then((latest_results: WordPrediction[]) => {
      console.log("Got results: ", latest_results);
      
      // Add the results to the operator
      const updatedOperators = [...operators];
      updatedOperators[updatedOperators.length - 1].results = latest_results;
      setOperators(updatedOperators);

      // setPredictions(latest_results);
      handleAddOperator();
    }).catch(e => console.log)
  };

  console.log("Predictions: ", predictions);
  console.log("Operators: ", operators);

  function resetApp(): void {
    console.log(initialAppState)
    setOperators(clone(initialAppState.ops));
    setPredictions(clone(initialAppState.predictions));
  }

  return (
    <div className={styles.container}>

      <main className={styles.main}>
        <h1 className={styles.title}>
          Tip of My Tongue
        </h1>

        <p className={styles.description}>
          It's 2023... why are we still using a thesaurus?
        </p>

        <div className={styles.card}>
            <h2>I'm thinking of a word like...</h2>

            {operators.map((operator, index) => (
              <div key={index} className={styles.operator}>
                  {/* Word Function Operator Dropdown */}
                  {index > 0 && (
                    <select
                    value={operator.function}
                    onChange={(e) => handleOperatorChange(index, 'function', e.target.value as 'more_like' | 'less_like')}
                  >
                    <option value="more_like">More like</option>
                    <option value="less_like">Less like</option>
                  </select>
                  )}

                  
                <div className={styles.operatorInputs}>
                  <input
                    type="text"
                    value={operator.description}
                    onChange={(e) => handleOperatorChange(index, 'description', e.target.value)}
                  />

                  {/* Remove Operator Button */}
                  {index > 0 && (
                    <button onClick={() => handleRemoveOperator(index)}>X</button>
                  )}
                </div>


                {/* Results */}
                {operator.results && (
                <div className={styles.operatorResults}>
                  {operator.results.map((prediction, index) => (
                    <button key={index}>
                      <span>{prediction.word}</span>
                    </button>
                  ))}
                </div> 
                )}
              </div>
            ))}

            <div className={styles.searchControls}>
              {/* <button onClick={handleAddOperator}>Add Operator</button> */}
              <button onClick={() => { calculateWords(operators) }}>Get Words</button>
              
              {/* Hide the reset button until an operator is provided */}
              {operators.length > 1 && (<button onClick={() => resetApp()}>Start Over</button>)}
            </div>
        </div>
      </main>

      <footer className={styles.footer}>
        <a href="#">See how it works</a>
      </footer>
    </div>
  )
}

export default App;
