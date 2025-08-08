"""
Advanced Sentiment-Performance Analysis Module
Deep dive analysis with statistical testing and modeling - EXTRA CAREFUL VERSION
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from scipy import stats as scipy_stats
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')
from typing import Dict, List, Tuple, Optional, Union

class DataValidator:
    """Comprehensive data validation utilities"""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_cols: List[str] = None) -> Dict[str, bool]:
        """Validate dataframe structure and content"""
        validation_results = {
            'is_valid': True,
            'is_empty': len(df) == 0,
            'has_nulls': df.isnull().any().any(),
            'column_issues': [],
            'data_type_issues': []
        }
        
        if validation_results['is_empty']:
            validation_results['is_valid'] = False
            return validation_results
        
        if required_cols:
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                validation_results['column_issues'] = missing_cols
                validation_results['is_valid'] = False
        
        return validation_results
    
    @staticmethod
    def safe_column_finder(df: pd.DataFrame, search_terms: List[str], exact_match: bool = False) -> List[str]:
        """Safely find columns containing search terms"""
        if df.empty:
            return []
        
        available_cols = list(df.columns)
        matching_cols = []
        
        for term in search_terms:
            for col in available_cols:
                if exact_match:
                    if term.lower() == col.lower():
                        matching_cols.append(col)
                else:
                    if term.lower() in col.lower():
                        matching_cols.append(col)
        
        return list(set(matching_cols))  # Remove duplicates

class StatisticalAnalyzer:
    """Advanced statistical analysis with robust error handling"""
    
    def __init__(self):
        self.validator = DataValidator()
    
    def safe_hypothesis_testing(self, df: pd.DataFrame, group_col: str, value_col: str) -> Dict[str, any]:
        """Perform hypothesis testing with comprehensive error handling"""
        print(f"📊 **HYPOTHESIS TESTING: {group_col} vs {value_col}**")
        print("=" * 60)
        
        # Validate inputs
        validation = self.validator.validate_dataframe(df, [group_col, value_col])
        if not validation['is_valid']:
            return {
                'error': 'Data validation failed',
                'details': validation
            }
        
        try:
            # Get unique groups
            groups = df[group_col].unique()
            print(f"   Groups found: {list(groups)}")
            
            if len(groups) < 2:
                return {
                    'error': 'Need at least 2 groups for comparison',
                    'groups_found': len(groups)
                }
            
            # Prepare data for each group
            group_data = {}
            for group in groups:
                group_values = df[df[group_col] == group][value_col].dropna()
                if len(group_values) > 0:
                    group_data[group] = group_values
                    print(f"   {group}: {len(group_values)} observations")
            
            if len(group_data) < 2:
                return {
                    'error': 'Insufficient data in groups',
                    'valid_groups': len(group_data)
                }
            
            # Perform statistical tests
            test_results = {
                'group_data': group_data,
                'descriptive_stats': {},
                'normality_tests': {},
                'variance_tests': {},
                'mean_comparison_tests': {}
            }
            
            # Descriptive statistics
            for group, data in group_data.items():
                test_results['descriptive_stats'][group] = {
                    'count': len(data),
                    'mean': float(data.mean()),
                    'median': float(data.median()),
                    'std': float(data.std()),
                    'min': float(data.min()),
                    'max': float(data.max())
                }
            
            # Normality tests (Shapiro-Wilk for smaller samples)
            for group, data in group_data.items():
                if len(data) >= 3:  # Minimum for Shapiro-Wilk
                    try:
                        if len(data) <= 5000:  # Shapiro-Wilk limitation
                            stat, p_value = stats.shapiro(data)
                            test_results['normality_tests'][group] = {
                                'test': 'Shapiro-Wilk',
                                'statistic': float(stat),
                                'p_value': float(p_value),
                                'is_normal': p_value > 0.05
                            }
                    except Exception as e:
                        test_results['normality_tests'][group] = {
                            'error': str(e)
                        }
            
            # Compare means between groups
            group_list = list(group_data.keys())
            if len(group_list) >= 2:
                # Pairwise comparisons
                for i, group1 in enumerate(group_list):
                    for j, group2 in enumerate(group_list[i+1:], i+1):
                        try:
                            data1 = group_data[group1]
                            data2 = group_data[group2]
                            
                            # T-test (assuming unequal variances)
                            t_stat, t_p = stats.ttest_ind(data1, data2, equal_var=False)
                            
                            # Mann-Whitney U test (non-parametric)
                            u_stat, u_p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
                            
                            comparison_key = f"{group1}_vs_{group2}"
                            test_results['mean_comparison_tests'][comparison_key] = {
                                't_test': {
                                    'statistic': float(t_stat),
                                    'p_value': float(t_p),
                                    'significant': t_p < 0.05
                                },
                                'mann_whitney': {
                                    'statistic': float(u_stat),
                                    'p_value': float(u_p),
                                    'significant': u_p < 0.05
                                }
                            }
                            
                        except Exception as e:
                            test_results['mean_comparison_tests'][f"{group1}_vs_{group2}"] = {
                                'error': str(e)
                            }
            
            print(f"✅ Hypothesis testing completed for {len(group_data)} groups")
            return test_results
            
        except Exception as e:
            print(f"❌ Statistical analysis error: {e}")
            return {'error': str(e)}
    
    def advanced_correlation_analysis(self, df: pd.DataFrame, target_col: str) -> Dict[str, any]:
        """Advanced correlation analysis with significance testing"""
        print(f"🔗 **ADVANCED CORRELATION ANALYSIS with {target_col}**")
        print("=" * 60)
        
        # Validate target column
        if target_col not in df.columns:
            available_targets = self.validator.safe_column_finder(df, ['pnl', 'profit', 'return'])
            print(f"⚠️ Target column '{target_col}' not found")
            print(f"   Available alternatives: {available_targets}")
            if available_targets:
                target_col = available_targets[0]
                print(f"   Using: {target_col}")
            else:
                return {'error': f'No suitable target column found'}
        
        try:
            # Get numeric columns for correlation
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if target_col not in numeric_cols:
                return {'error': f'Target column {target_col} is not numeric'}
            
            # Remove target from predictors
            predictor_cols = [col for col in numeric_cols if col != target_col]
            
            if len(predictor_cols) == 0:
                return {'error': 'No numeric predictor columns found'}
            
            correlation_results = {
                'target_column': target_col,
                'predictor_columns': predictor_cols,
                'correlations': {},
                'significant_correlations': []
            }
            
            target_data = df[target_col].dropna()
            
            # Calculate correlations with significance tests
            for predictor in predictor_cols:
                try:
                    predictor_data = df[predictor].dropna()
                    
                    # Ensure we have overlapping data
                    common_indices = target_data.index.intersection(predictor_data.index)
                    if len(common_indices) < 3:
                        continue
                    
                    target_aligned = target_data.loc[common_indices]
                    predictor_aligned = predictor_data.loc[common_indices]
                    
                    # Pearson correlation
                    pearson_r, pearson_p = stats.pearsonr(target_aligned, predictor_aligned)
                    
                    # Spearman correlation (rank-based)
                    spearman_r, spearman_p = stats.spearmanr(target_aligned, predictor_aligned)
                    
                    correlation_results['correlations'][predictor] = {
                        'pearson': {
                            'correlation': float(pearson_r),
                            'p_value': float(pearson_p),
                            'significant': pearson_p < 0.05
                        },
                        'spearman': {
                            'correlation': float(spearman_r),
                            'p_value': float(spearman_p),
                            'significant': spearman_p < 0.05
                        },
                        'sample_size': len(common_indices)
                    }
                    
                    # Track significant correlations
                    if pearson_p < 0.05 and abs(pearson_r) > 0.1:
                        correlation_results['significant_correlations'].append({
                            'predictor': predictor,
                            'correlation': float(pearson_r),
                            'p_value': float(pearson_p),
                            'strength': 'Strong' if abs(pearson_r) > 0.5 else 'Moderate' if abs(pearson_r) > 0.3 else 'Weak'
                        })
                        
                except Exception as e:
                    correlation_results['correlations'][predictor] = {'error': str(e)}
            
            # Sort significant correlations by strength
            correlation_results['significant_correlations'].sort(
                key=lambda x: abs(x['correlation']), reverse=True
            )
            
            print(f"✅ Analyzed correlations for {len(predictor_cols)} predictors")
            print(f"   Significant correlations found: {len(correlation_results['significant_correlations'])}")
            
            return correlation_results
            
        except Exception as e:
            print(f"❌ Correlation analysis error: {e}")
            return {'error': str(e)}

class PredictiveModeler:
    """Machine learning models for sentiment-based performance prediction"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.models = {}
        self.scalers = {}
        self.encoders = {}
    
    def safe_model_preparation(self, df: pd.DataFrame, target_col: str, 
                              feature_cols: List[str] = None) -> Dict[str, any]:
        """Safely prepare data for machine learning"""
        print(f"🤖 **PREPARING ML MODEL DATA**")
        print("=" * 50)
        
        # Validate inputs
        validation = self.validator.validate_dataframe(df, [target_col])
        if not validation['is_valid']:
            return {'error': 'Data validation failed', 'details': validation}
        
        try:
            # Determine feature columns
            if feature_cols is None:
                # Automatically select numeric features
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                feature_cols = [col for col in numeric_cols if col != target_col]
                
                # Add categorical features
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                feature_cols.extend(categorical_cols)
            
            # Validate feature columns exist
            available_features = [col for col in feature_cols if col in df.columns]
            if len(available_features) == 0:
                return {'error': 'No valid feature columns found'}
            
            print(f"   Target: {target_col}")
            print(f"   Features: {len(available_features)} columns")
            
            # Prepare dataset
            model_data = df[available_features + [target_col]].copy()
            
            # Remove rows with missing target
            model_data = model_data.dropna(subset=[target_col])
            
            if len(model_data) == 0:
                return {'error': 'No data remaining after removing missing targets'}
            
            # Separate features and target
            X = model_data[available_features].copy()
            y = model_data[target_col].copy()
            
            # Handle categorical variables
            categorical_features = X.select_dtypes(include=['object']).columns.tolist()
            numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
            
            print(f"   Categorical features: {len(categorical_features)}")
            print(f"   Numeric features: {len(numeric_features)}")
            
            # Encode categorical variables
            for col in categorical_features:
                try:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col].astype(str))
                    self.encoders[col] = le
                except Exception as e:
                    print(f"⚠️ Could not encode {col}: {e}")
                    X = X.drop(columns=[col])
            
            # Handle missing values in features
            X = X.fillna(X.mean() if len(numeric_features) > 0 else 0)
            
            # Create binary classification target if continuous
            if y.dtype in ['float64', 'int64'] and y.nunique() > 10:
                # Convert to binary: profitable vs not profitable
                y_binary = (y > 0).astype(int)
                target_type = 'binary_classification'
                print(f"   Target converted to binary: {y_binary.sum()} positive, {len(y_binary) - y_binary.sum()} negative")
            else:
                y_binary = y
                target_type = 'multiclass' if y.nunique() > 2 else 'binary'
                print(f"   Target type: {target_type}, {y.nunique()} classes")
            
            preparation_results = {
                'X': X,
                'y': y_binary,
                'original_target': y,
                'feature_columns': list(X.columns),
                'target_column': target_col,
                'target_type': target_type,
                'sample_size': len(X),
                'feature_count': len(X.columns),
                'categorical_features': categorical_features,
                'numeric_features': numeric_features
            }
            
            print(f"✅ Model data prepared: {len(X)} samples, {len(X.columns)} features")
            return preparation_results
            
        except Exception as e:
            print(f"❌ Model preparation error: {e}")
            return {'error': str(e)}
    
    def safe_model_training(self, model_data: Dict[str, any]) -> Dict[str, any]:
        """Train ML models with comprehensive error handling"""
        print(f"\n🎯 **TRAINING MACHINE LEARNING MODELS**")
        print("=" * 50)
        
        if 'error' in model_data:
            return model_data
        
        try:
            X = model_data['X']
            y = model_data['y']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y if len(y.unique()) > 1 else None
            )
            
            print(f"   Training set: {len(X_train)} samples")
            print(f"   Test set: {len(X_test)} samples")
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            self.scalers['main'] = scaler
            
            training_results = {
                'data_split': {
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                },
                'models': {},
                'feature_importance': {}
            }
            
            # Train Random Forest
            try:
                print("   Training Random Forest...")
                rf_model = RandomForestClassifier(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2
                )
                rf_model.fit(X_train, y_train)
                
                # Predictions
                rf_train_pred = rf_model.predict(X_train)
                rf_test_pred = rf_model.predict(X_test)
                
                # Metrics
                training_results['models']['random_forest'] = {
                    'train_accuracy': float(accuracy_score(y_train, rf_train_pred)),
                    'test_accuracy': float(accuracy_score(y_test, rf_test_pred)),
                    'feature_importance': dict(zip(X.columns, rf_model.feature_importances_))
                }
                
                # Feature importance
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf_model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                training_results['feature_importance']['random_forest'] = feature_importance.to_dict('records')
                
                self.models['random_forest'] = rf_model
                print("   ✅ Random Forest trained successfully")
                
            except Exception as e:
                print(f"   ⚠️ Random Forest training failed: {e}")
                training_results['models']['random_forest'] = {'error': str(e)}
            
            # Train Logistic Regression
            try:
                print("   Training Logistic Regression...")
                lr_model = LogisticRegression(
                    random_state=42, 
                    max_iter=1000,
                    solver='liblinear'
                )
                lr_model.fit(X_train_scaled, y_train)
                
                # Predictions
                lr_train_pred = lr_model.predict(X_train_scaled)
                lr_test_pred = lr_model.predict(X_test_scaled)
                
                # Metrics
                training_results['models']['logistic_regression'] = {
                    'train_accuracy': float(accuracy_score(y_train, lr_train_pred)),
                    'test_accuracy': float(accuracy_score(y_test, lr_test_pred)),
                    'coefficients': dict(zip(X.columns, lr_model.coef_[0] if len(lr_model.coef_.shape) > 1 else lr_model.coef_))
                }
                
                self.models['logistic_regression'] = lr_model
                print("   ✅ Logistic Regression trained successfully")
                
            except Exception as e:
                print(f"   ⚠️ Logistic Regression training failed: {e}")
                training_results['models']['logistic_regression'] = {'error': str(e)}
            
            # Store test data for future use
            training_results['test_data'] = {
                'X_test': X_test,
                'y_test': y_test,
                'X_test_scaled': X_test_scaled
            }
            
            successful_models = [name for name, results in training_results['models'].items() 
                               if 'error' not in results]
            
            print(f"✅ Training completed: {len(successful_models)} models trained successfully")
            return training_results
            
        except Exception as e:
            print(f"❌ Model training error: {e}")
            return {'error': str(e)}

class AdvancedAnalyzer:
    """Main orchestrator for advanced sentiment-performance analysis"""
    
    def __init__(self):
        self.statistical_analyzer = StatisticalAnalyzer()
        self.predictive_modeler = PredictiveModeler()
        self.validator = DataValidator()
    
    def comprehensive_advanced_analysis(self, master_df: pd.DataFrame) -> Dict[str, any]:
        """Perform comprehensive advanced analysis with maximum error handling"""
        print("🎯 **COMPREHENSIVE ADVANCED SENTIMENT-PERFORMANCE ANALYSIS**")
        print("=" * 80)
        
        # Validate input
        validation = self.validator.validate_dataframe(master_df)
        if not validation['is_valid']:
            return {'error': 'Input validation failed', 'details': validation}
        
        print(f"📊 Input dataset: {master_df.shape[0]:,} records, {master_df.shape[1]} features")
        
        analysis_results = {
            'dataset_info': {
                'shape': master_df.shape,
                'columns': list(master_df.columns),
                'memory_usage_mb': master_df.memory_usage(deep=True).sum() / 1024**2
            }
        }
        
        # 1. Advanced Statistical Analysis
        try:
            print("\n" + "="*60)
            print("1. ADVANCED STATISTICAL ANALYSIS")
            print("="*60)
            
            # Find regime and PnL columns
            regime_cols = self.validator.safe_column_finder(master_df, ['regime', 'market'])
            pnl_cols = self.validator.safe_column_finder(master_df, ['pnl', 'profit', 'return'])
            
            if regime_cols and pnl_cols:
                hypothesis_results = self.statistical_analyzer.safe_hypothesis_testing(
                    master_df, regime_cols[0], pnl_cols[0]
                )
                analysis_results['statistical_analysis'] = hypothesis_results
            else:
                analysis_results['statistical_analysis'] = {
                    'error': f'Required columns not found. Regime: {regime_cols}, PnL: {pnl_cols}'
                }
                
        except Exception as e:
            analysis_results['statistical_analysis'] = {'error': str(e)}
        
        # 2. Advanced Correlation Analysis
        try:
            print("\n" + "="*60)
            print("2. ADVANCED CORRELATION ANALYSIS")
            print("="*60)
            
            pnl_cols = self.validator.safe_column_finder(master_df, ['pnl', 'profit', 'return'])
            
            if pnl_cols:
                correlation_results = self.statistical_analyzer.advanced_correlation_analysis(
                    master_df, pnl_cols[0]
                )
                analysis_results['correlation_analysis'] = correlation_results
            else:
                analysis_results['correlation_analysis'] = {
                    'error': 'No suitable target column found for correlation analysis'
                }
                
        except Exception as e:
            analysis_results['correlation_analysis'] = {'error': str(e)}
        
        # 3. Predictive Modeling
        try:
            print("\n" + "="*60)
            print("3. PREDICTIVE MODELING")
            print("="*60)
            
            pnl_cols = self.validator.safe_column_finder(master_df, ['pnl', 'profit', 'return'])
            
            if pnl_cols:
                # Prepare model data
                model_data = self.predictive_modeler.safe_model_preparation(
                    master_df, pnl_cols[0]
                )
                
                if 'error' not in model_data:
                    # Train models
                    training_results = self.predictive_modeler.safe_model_training(model_data)
                    analysis_results['predictive_modeling'] = {
                        'data_preparation': model_data,
                        'training_results': training_results
                    }
                else:
                    analysis_results['predictive_modeling'] = model_data
            else:
                analysis_results['predictive_modeling'] = {
                    'error': 'No suitable target column found for predictive modeling'
                }
                
        except Exception as e:
            analysis_results['predictive_modeling'] = {'error': str(e)}
        
        # Summary
        successful_analyses = sum(1 for key, value in analysis_results.items() 
                                if key != 'dataset_info' and 'error' not in str(value))
        total_analyses = len(analysis_results) - 1  # Exclude dataset_info
        
        analysis_results['summary'] = {
            'total_analyses': total_analyses,
            'successful_analyses': successful_analyses,
            'success_rate': successful_analyses / total_analyses if total_analyses > 0 else 0
        }
        
        print(f"\n✅ **ADVANCED ANALYSIS COMPLETE**")
        print(f"   Success rate: {successful_analyses}/{total_analyses} ({analysis_results['summary']['success_rate']:.1%})")
        
        return analysis_results

if __name__ == "__main__":
    print("🛠️ **Web3 Trading Analysis - Advanced Analysis Module Loaded**")
    print("Available classes: StatisticalAnalyzer, PredictiveModeler, AdvancedAnalyzer")
