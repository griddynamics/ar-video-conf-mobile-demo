import React from 'react';
import ArVideoConf from './components/ArVideoConf';
import Header from './components/Header';
import './styles.scss';

const App = () => {
  return (
    <div className="pageWrap">
      <div className="pageContent"> 
        <Header/>
        <ArVideoConf/>
      </div>
    </div>
  );
};

export default App;
