# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
#import matplotlib as m
class sample_generator():
    
    def __init__(self):
        super(sample_generator,self).__init__()
    
    def generate_sample(self,dim,n,low=0,high=1):
        """
        returns a sample using 
        np.random.uniform, for other distributions feel free to modify this function.
        """
        return np.random.uniform(low=low,high=high,size=[n,dim])
    
    def linearly_separated_target(self,sample,params):
        """
            Given a sample it linearly separates the sampe in two regions:
            The sample should have N columns, and the parameters should have N+1 numbers.
             The separation works as follows:
                 
                 bias=parameters[0]
                 
                 bias+parameters[1:]*sample>=0 --> 1
                 bias+parameters[1:]*sample<0  --> 0
                 
                 i.e.
                 sample=np.random.uniform(low=0,high=10,size=[10,2])
                 params=np.array([2,-2,2])
                 
                 res=(np.matmul(formatted_sample,params)>=0)*1
                 
            The objective of this function is not efficiency but informative. 
            There are better ways to do this.     
        """
#        n_cols=sample.shape[1]
        n_rows=sample.shape[0]
        bias_column=np.ones(shape=(n_rows,1))
        formatted_sample=np.hstack((bias_column,sample))
        
        if len(params)!=formatted_sample.shape[1]:
            return None
        res=(np.matmul(formatted_sample,params)>=0)*1   
#       res=(np.matmul(formatted_sample,params)>=0).astype(int) #Other ways to do it
#       res=[int(x>=0) for x in res]  #Other ways to do it
        return res
    
    def polinomically_separated_target(self,sample,params):
        """
            Given a sample it separates it using parameters in form of a list of tuples:
                params=[(column_ix, coeff,power)]
            
            To add a bias use column_ix=-1  
            
            sample=np.random.uniform(low=0,high=10,size=[10,2])
            params=[
                    (0,2,1),
                    (1,2,1),
                    (-1,2,1)
                    ]
        
        """
        params.sort()
        n_rows=sample.shape[0]
        y=np.zeros(shape=(n_rows,1))
        old_col=None
        for param_tuple in params:
            col  = param_tuple[0]
            coeff= param_tuple[1]
            power= param_tuple[2]
            
            if old_col!=col and col!=-1:
                column=sample[:,col]
            elif col==-1:
                column=np.ones(shape=(n_rows,1))

            y=y+ coeff*np.power(column, power)
            
        return (y>=0)*1
        
    def visualize_sample(self,sample2D,target=None,params=None,label_data=False):
        
        fig =plt.figure(figsize=(9, 9))
        fig.show()
        c=None
        x=sample2D[:,0]
        y=sample2D[:,1]
        if not target is None:
            c=target

        plt.scatter(x, y, c=c,cmap='rainbow')
        if not params is None:
            """ For now only linear parameters are supported y= m*x+b"""
            
            if params[2]==0:
                if params[1]==0:
                    print("Both parameters cant be 0, Remember p0+p1*x+p2*y=0.")
                    return None
                b=params[0]
                yl=[np.floor(min(y)),np.ceil(max(y))]
                xl=[-b/params[1],-b/params[1]]
            else:
                b=params[0]/params[2]
                m=params[1]/params[2]
                xl=[np.floor(min(x)),np.ceil(max(x))]
                yl=[-b- m*xl[0],-b-m*xl[1]]
            plt.plot(xl,yl)
            plt.axis(
                        (
                         np.floor(min(x)), np.ceil(max(x)),
                         np.floor(min(y)), np.ceil(max(y))
                         )
                     )
        if label_data and (not target is None):
            for i, txt in enumerate(target):
                plt.annotate(str(txt), (x[i], y[i]))
            
        return None
                    
        
    def get_zeros_in_interval(x_lims=(0,1),y_lims=(0,1),params=None,threshold=0.001):
        """
            Given an interval it solves for all the points that are 0 given a 
            polynomial in 2 variables x and y.
            Line eqn: 2-2x+2y=0
            params=[
            (0,2,1),
            (1,2,1),
            (-1,2,1)
            ]
        
        """
        
        
        return None