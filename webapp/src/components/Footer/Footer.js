import React from 'react';
import { footer, legal } from './Footer.module.scss';

const currentYear = new Date().getFullYear();

const Footer = () => {
  return (
    <div className={footer}>
      <span className={legal}>© {currentYear} Grid Dynamics Labs</span>
    </div>
  )
}

export default Footer;