import './theme.css';
import { AppShell } from './app-shell';

const root = document.getElementById('app');
if (!root) {
  throw new Error("Missing #app root element — check index.html");
}

new AppShell(root).init();
