import React from 'react';
import Logo from '../../assets/logo.png';
import { header, demoLink, title, logoContainer, logoImg } from './Header.module.scss';

const Header = () => {
  const DEMO_PORTAL_URL = "http://demo.griddynamics.net";

  return (
    <div className={header}>
        <div className={logoContainer}>
          <a href={DEMO_PORTAL_URL}>
            <img src={Logo} className={logoImg} alt="Grid Labs Demo" />
          </a>
          <div className={title}>
              <span>AR for Video Conferencing</span>
          </div>
        </div>
        <div>
          <a className={demoLink} href={DEMO_PORTAL_URL}>All Demos</a>
        </div>
    </div>
  )
};

export default Header;