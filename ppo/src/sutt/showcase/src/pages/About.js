import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

import '../App.css';
import '../styles/About.css'

export default function About() {
  const { pathname } = useLocation();

  useEffect(() => {
    // Scroll to top when the route changes
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [pathname]);

  return (
    <>
    <div className="about-container">
        <h1>About</h1>
        <p>Welcome to the About page, where we unveil the mission in the fight against hunger and malnutrition worldwide.</p>
        <p>Our journey begins within the realm of the Food and Agriculture Organization of the United Nations (FAO), a distinguished agency committed to fostering a world free from hunger. Our primary objective is to delve into a comprehensive study on global undernutrition.</p>
        <p>Before embarking on this crucial mission, our team diligently delves into the latest literature surrounding undernutrition, ensuring that our approach is not only informed but also aligned with the most recent advancements in the field. This initial step sets the stage for our subsequent endeavors to make a tangible impact.
        </p>
        <p>
        A cornerstone of our efforts involves conducting a meticulous statistical study. This study is not merely a collection of numbers but a strategic tool aimed at identifying countries in dire need. By scrutinizing and interpreting the data, we gain insights into the multifaceted causes of hunger that plague communities worldwide.
        </p>
        <p>Let's take a closer look at the roadmap we've charted for ourselves - a structured workflow designed to navigate the complexities of our mission:</p>
        <ol>
        <li>
          <b>Data collection</b> : We begin by gathering a wealth of information, ensuring our dataset is comprehensive and reflective of the global landscape of undernutrition. The data we used in this website come from the Food and Agriculture Organization of the United Nations (FAO), in this <a href="https://www.fao.org/gift-individual-food-consumption/data/en">page</a>.
          </li>
          <li>
            <b>Data discovery</b>: Armed with our data, we embark on a journey of exploration, seeking patterns, anomalies, and hidden gems that will shape our understanding.
          </li>
          <li>
            <b>Data cleaning</b>: With a keen eye for precision, we sift through the data, ensuring its integrity and reliability for a robust analysis.
          </li>
          <li>
            <b>Computing new variables</b>: We don't just stop at data; we transform it. By computing new variables, we lay the groundwork for a nuanced and insightful analysis.
          </li>
          <li>
            <b>Identifying major trends</b>: Our analytical prowess comes to the forefront as we identify major trends, unraveling the complex web of factors contributing to undernutrition.
          </li>
          <li>
            <b>Communicating our analysis</b>: Finally, armed with knowledge and insights, we take on the responsibility of communicating our findings to the world. Through reports, visualizations, and impactful narratives, we strive to bring attention to the urgency of addressing hunger and malnutrition in the present website.
          </li>
        </ol>
        <p>
        Join us on this transformative journey as we strive to create a world where no one goes to bed hungry. Together, let's turn the tide against undernutrition and pave the way for a brighter, nourished future for all.
        </p>       
    </div>
    </>
  );
}