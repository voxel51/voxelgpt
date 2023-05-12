import { useState, useEffect } from 'react';

const useTypewriterEffect = (fullText, speed) => {
  const [index, setIndex] = useState(0);

  fullText = fullText || '';

  // create an interval that increments the index by 1
  // the interval is cleared when the index is equal to the length of the fullText
  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prevIndex) => prevIndex + 1);
    }, speed);

    if (index === fullText.length) {
      clearInterval(interval);
    }
    
    return () => clearInterval(interval);
  }, [index, fullText.length, speed]);
  
  return fullText.slice(0, index);
};

export default useTypewriterEffect;