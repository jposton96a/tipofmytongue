import React, { useState } from 'react';
import logo from './logo.svg';

import styles from './Home.module.css'

import Button from '@mui/joy/Button';
import Input from '@mui/joy/Input';
import Select from '@mui/joy/Select';
import Option from '@mui/joy/Option';
import Divider from '@mui/joy/Divider';
import Switch from '@mui/joy/Switch';
import Chip from '@mui/joy/Chip';
import Checkbox from '@mui/joy/Checkbox';

import { FiAlertCircle, FiSlash } from "react-icons/fi";
import { FiSearch } from "react-icons/fi";
import { FiThumbsUp } from "react-icons/fi";
import { FiThumbsDown } from "react-icons/fi";
import { FiCheckCircle } from "react-icons/fi";
import { Accordion, AccordionDetails, AccordionGroup, AccordionSummary, LinearProgress, Typography } from '@mui/joy';
import { isDisabled } from '@testing-library/user-event/dist/utils';


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


const PredictionChip = (props: {displayText: string}) => {
  const [checked, setChecked] = useState(false);

  return (
    <Chip
      sx={{margin: "4px"}}
      key={props.displayText}
      variant="plain"
      color={checked ? 'primary' : 'neutral'}
      startDecorator={
        checked && <FiCheckCircle style={{ zIndex: 1, pointerEvents: 'none' }} />
      }
    >
      <Checkbox
        variant="outlined"
        color={checked ? 'primary' : 'neutral'}
        disableIcon
        overlay
        label={props.displayText}
        checked={checked}
        onChange={(event) => {
          // setSelected((names) =>
          //   !event.target.checked
          //     ? names.filter((n) => n !== name)
          //     : [...names, name],
          // );
        }}
      />
    </Chip>
  )
}

const OperatorInput = (props: {index: number, onSubmit: any, operation: WordOperation, updateOperator:any, disabled: boolean}) => {
  const opFunctionBool = props.operation.function != "less_like";
  
  const placeholder = {
    "more_like": "Words more like ...",
    "less_like": "Words less like ...",
    "start": "I'm thinking of a word like..."
  }[props.operation.function]

  return (
    <Input
      autoFocus={true}
      required={true}
      value={props.operation.description}
      disabled={props.disabled}
      onChange={(e) => props.updateOperator(props.index, 'description', e.target.value)}
      placeholder={placeholder}
      startDecorator={(<React.Fragment>
          {props.disabled ? (
            opFunctionBool ? (
              <FiThumbsUp style={{ color: '#0B6BCB' }} />
            ) : (
              <FiThumbsDown style={{ color: '#A51818' }} />
            )
          ) : (
            <Switch
              disabled={props.disabled}
              color={opFunctionBool ?   'primary' : 'danger'}
              startDecorator={
                <FiThumbsDown
                  style={{ color: opFunctionBool ? '#555E68' : '#A51818' }}
                />
              }
              endDecorator={
                <FiThumbsUp style={{ color: opFunctionBool ? '#0B6BCB' : '#555E68' }} />
              }
              checked={opFunctionBool}
              onChange={(event: React.ChangeEvent<HTMLInputElement>) =>
                props.updateOperator(props.index, 'function', event.target.checked ? 'more_like' : 'less_like')
              }
            />
          )}
          <Divider sx={{marginLeft: "10px"}} orientation="vertical" />
        </React.Fragment>
        )
      }
      // endDecorator={!props.disabled && (<FiSearch></FiSearch>)}
      // endDecorator={!props.disabled && (<Button onClick={props.onSubmit}><FiSearch></FiSearch></Button>)}
    />
  )
}


function App() {
  const [operators, setOperators] = useState<WordOperation[]>(clone(initialAppState.ops));
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(false);

  const handleAddOperator = () => {
    setOperators([...operators, { description: '', function: 'more_like' }]);
  };

  const handleOperatorChange = (index: number, field: keyof WordOperation, value: any) => {
    const updatedOperators = [...operators];
    updatedOperators[index][field] = value;
    setOperators(updatedOperators);
  };

  const handleRemoveOperator = (index: number) => {
    const updatedOperators = [...operators];
    updatedOperators.splice(index, 1);
    setOperators(updatedOperators);
  };

  const resetApp = () => {
    setOperators(clone(initialAppState.ops));
  }

  const calculateWords = (operators: WordOperation[]) => {
    operators = operators.filter(operator => operator.description !== "" && operator.description !== null);
    
    setLoading(true);
    setError(false);


    getWords(operators).then((latest_results: WordPrediction[]) => {
      console.log("Got results: ", latest_results);
      
      // Add the results to the operator
      const updatedOperators = [...operators];
      updatedOperators[updatedOperators.length - 1].results = latest_results;
      setOperators(updatedOperators);

      // setPredictions(latest_results);
      handleAddOperator();
    }).catch(e => {
      console.log(e);
      setError(true); 
    }).finally(() => setLoading(false))
  };

  // Debug:
  console.log("Loading: ", loading);
  console.log("Operators: ", operators);

  return (
    <div className={styles.container}>

      <main className={styles.main}>
        <Typography level="h1" fontWeight={"md"} color="neutral" >
          TipofMyTongue.ai
        </Typography>

        <Typography level="h4" fontWeight={"sm"} color="neutral" padding={"10px"} sx={{textAlign: "center"}}>
          It's 2023... why are we still using a thesaurus?
        </Typography>

        <div className={styles.card}>

          <form style={{display: "flex", flexDirection: "column", alignItems: "center"}} onSubmit={(event) => {
            event.preventDefault();
            calculateWords(operators);
          }}>

            <AccordionGroup size="sm" disableDivider>
              {operators.map((operator, index) => (
                <Accordion 
                  key={index} 
                  className={styles.operator} 
                  expanded={index == operators.length - 2}>

                  <AccordionSummary className={styles.opGroup}>
                    <OperatorInput 
                      index={index}
                      operation={operator}
                      updateOperator={handleOperatorChange}
                      disabled={index != operators.length - 1}
                      onSubmit={() => { calculateWords(operators) }}></OperatorInput>
                  </AccordionSummary>

                  {/* Results */}
                  <AccordionDetails>
                    {operator.results && (
                      <div className={styles.operatorResults}>
                        {operator.results.map((prediction, index) => (
                          <PredictionChip key={prediction.word} displayText={prediction.word}></PredictionChip>
                          ))}
                      </div> 
                    )}
                </AccordionDetails>
                
                </Accordion>
              ))}
              </AccordionGroup>
            
              <div style={{width: "50%", padding: "20px"}}>
                {loading && (<LinearProgress color='primary'/>)}
              </div>

              <div className={styles.searchControls}>
                {/* Hide the reset button until an operator is provided */}
                <Button type="submit" endDecorator={(<FiSearch></FiSearch>)}>Search</Button>
                {operators.length > 1 && (<Button color='danger' variant='outlined' onClick={() => resetApp()} endDecorator={(<FiSlash></FiSlash>)}>Start Over</Button>)}
              </div>
            </form>
        </div>
      </main>

      <footer className={styles.footer}>
        <a href="#">See how it works</a>
      </footer>
    </div>
  )
}

export default App;

{/* <Select
variant="plain"
value={"more_like"}
onChange={(_, value) => { console.log(value);}}
slotProps={{
listbox: {
variant: 'outlined',
},
}}
sx={{ mr: -1.5, '&:hover': { bgcolor: 'transparent' } }}
>
<Option value="more_like">More Like</Option>
<Option value="less_like">Less Like</Option>
</Select> */}