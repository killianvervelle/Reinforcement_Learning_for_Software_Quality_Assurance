import React, { useEffect,useState } from 'react';

import { useLocation } from 'react-router-dom';
import WorldMap from '../components/WorldMap';
import Legend from '../components/Legend';

import '../App.css';
import '../styles/WorldMap.css'
import '../styles/Legends.css'

const PopupComponent = () => {
  const [showPopup, setShowPopup] = useState(false);
  localStorage.clear()
  useEffect(() => {
    console.log('PopupComponent mounted');
    const hasPopupBeenShown = localStorage.getItem('popupShown');

    if (!hasPopupBeenShown) {
      localStorage.setItem('popupShown', 'true');
      setShowPopup(true);
    }
  }, []);

  const handlePopupClose = () => {
    console.log('Popup closed');
    setShowPopup(false);
    localStorage.removeItem('popupShown');
  };

  useEffect(() => {
    const closePopupOnClick = (event) => {
        handlePopupClose();
    };
    document.addEventListener('click', closePopupOnClick);
    return () => {
      document.removeEventListener('click', closePopupOnClick);
    };
  }, [showPopup]);

  return (
    <>
         {showPopup && (
          
        <div className="welcome" onClick={handlePopupClose}>
          <button className="close-button" onClick={handlePopupClose} style={{fontSize: 23}}>
          X
        </button>
        <p><strong>Greetings newcomer and welcome,</strong></p>
        <p>We're thrilled to have you on board. To ensure you make the most out of your experience, we'd like to guide you through the options available to you:</p>
        <p><strong>Option 1:</strong> Select a <strong>Country</strong><br/>
        To tailor your experience and provide you with relevant information, please take a moment to select a country. This will grant you access to a variety of 
        data related to the country's distribution of food. Feel free to explore and interact with the stacked bar chart to delve into the details of each food utilization category.</p>
        <p><strong>Option 2:</strong> Explore the <strong>Charts</strong> Section<br/>
        To gain a comprehensive insight into the global food crisis and its root causes, delve into the facts and statistics presented in the Charts section. Feel free to customize the graph settings to observe trends over a span of 5 years.</p>
        <p>Thank you for using our application. We hope you enjoy your journey with us!</p>
        </div>
 )}
    </>
  );
};

const Home = () => {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [pathname]);

  return (
    <div className='home-container'>
      <div className="context-container">
        <h3> Context of the study </h3>
          <p>Malnutrition remains a critical global challenge, with millions of people experiencing inadequate access to 
          sufficient and nutritious food. The following equation, <strong>Domestic supply quantity = Production quantity + 
          Imports + Opening stock - Exports - Closing stock - Food - Feed - Seed - Losses -+ Processed - Others uses - 
          Tourist consumption - Residuals</strong>, highlights the complex dynamics involved in assessing the food supply chain. 
          Understanding this equation is crucial for comprehending the factors contributing to malnutrition, as it 
          encompasses production, trade, and stock variations. Analyzing each component — such as food loss, processing, 
          and diverse uses — provides valuable insights into the allocation of available resources and the identification 
          of potential areas for intervention in addressing malnutrition on a global scale.</p>
      </div>
      <div className='main-map-container'>
        <WorldMap />
        <Legend />
      </div>
      <PopupComponent />
    </div>
  );
};

export default Home;