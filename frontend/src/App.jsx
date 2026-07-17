import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ThemeProvider } from './components/ThemeProvider';
import Dashboard from './pages/Dashboard';
import CallDetail from './pages/CallDetail';

export default function App() {
  return (
    <ThemeProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/calls/:callId" element={<CallDetail />} />
        </Routes>
      </BrowserRouter>
    </ThemeProvider>
  );
}
