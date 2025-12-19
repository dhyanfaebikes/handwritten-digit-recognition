import { Routes } from '@angular/router';

import { HomePage } from './pages/home/home.page';
import { ComparisonPage } from './pages/comparison/comparison.page';

export const routes: Routes = [
  { path: '', component: HomePage },
  { path: 'comparison', component: ComparisonPage },
  { path: '**', redirectTo: '' }
];
