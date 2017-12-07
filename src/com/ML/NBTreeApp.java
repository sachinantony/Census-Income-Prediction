package com.ML;

import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import weka.classifiers.trees.NBTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.StringToWordVector;

public class NBTreeApp {

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		String filename = "Census.arff";
		try {
			DataSource source = new DataSource(filename);
			Instances instances = source.getDataSet();
			instances.setClassIndex(instances.numAttributes() - 1);
			int seed = instances.size();
			Random random = new Random(System.nanoTime());
			instances.randomize(random);
			int testSize = seed/4;
			int trainSize = seed - testSize;
			Instances trainInst = new Instances(instances, 0, trainSize);
			Instances testInst = new Instances(instances,trainSize,testSize);
			NBTree tree = new NBTree();
			
			tree.buildClassifier(trainInst);			
			StringToWordVector wv = new StringToWordVector();
			List<Double> predictionList = new ArrayList<Double>();
			List<Double> OriginalList = new ArrayList<Double>();	
		
			for(int i = 0; i < testSize; i++)
			{
				predictionList.add(tree.classifyInstance(testInst.get(i)));
				OriginalList.add(Double.parseDouble(testInst.instance(i).stringValue(14)));
				
			}

			System.out.println("Predicted Values: ");
			System.out.println(predictionList);
			System.out.println("Actual Values: ");
			System.out.println(OriginalList);
			
		int tp, tn, fp,fn;
		tp=0;tn=0;fp=0;fn=0;
		for(int i = 0; i < testSize; i++)
		{
			double o = OriginalList.get(i);
			double p = predictionList.get(i);
			
			if(o==1.0)
			{
				
				if(o==p)
					tp = tp+1;
				else
					fn = fn+1;
			}
			if(o==0.0)
			{
				if(o==p)
					tn = tn+1;
				else
					fp = fp+1;
			}
		}


		double numer = tp + tn;
			double PredAccuracy;
			PredAccuracy = (numer/testSize)*100;
			DecimalFormat numberFormat = new DecimalFormat("#.00");
			System.out.println("Accuracy: ");
			System.out.println(numberFormat.format(PredAccuracy)+"%");
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

	}

}
