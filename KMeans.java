import java.io.BufferedReader;
import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URI;
import java.util.Scanner;
import java.util.Vector;
import java.util.HashMap;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.WritableComparable;
import org.apache.hadoop.io.Writable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.mapreduce.lib.output.TextOutputFormat;

class Point implements WritableComparable {
	public double x;
	public double y;

	public void readFields(DataInput in) throws IOException {
		this.x = in.readDouble();
		this.y = in.readDouble();
	}

	public void write(DataOutput out) throws IOException {
		out.writeDouble(this.x);
		out.writeDouble(this.y);
	}

	public int compareTo(Object o) {
		/*
		 * How do you compare 2 Points? Compare the x components first; if equal,
		 * compare the y components.
		 */
		Point p = (Point) o;
		int xEquality = Double.compare(this.x, p.x);
		if (xEquality != 0) {
			return xEquality;
		}
		return Double.compare(this.y, p.y);
	}

	@Override
	public String toString() {
		return x + "," + y;
	}
}

class Avg implements Writable {
    public double sumX;
    public double sumY;
    public long count;
    
    Avg() {
    	this.sumX = 0.0;
    	this.sumY = 0.0;
    	this.count = 0;
    }
    
    Avg(double x, double y, long count) {
    	this.sumX = x;
    	this.sumY = y;
    	this.count = count;
    }
    
	public void readFields(DataInput in) throws IOException {
		this.sumX = in.readDouble();
		this.sumY = in.readDouble();
		this.count = in.readLong();
	}

	public void write(DataOutput out) throws IOException {
		out.writeDouble(this.sumX);
		out.writeDouble(this.sumY);
		out.writeLong(this.count);
	}

    @Override
    public String toString() {
    	return sumX + "," + sumY + "," + count;
    }
}

public class KMeans {
	static Vector<Point> centroids = new Vector<Point>(100);
	static HashMap<Point, Avg> table;

	public static class AvgMapper extends Mapper<Object, Text, Point, Avg> {

		@Override
		protected void setup(Context context)
				throws IOException, InterruptedException {
			super.setup(context);

			URI[] paths = context.getCacheFiles();
			Configuration conf = context.getConfiguration();
			FileSystem fs = FileSystem.get(conf);
			BufferedReader reader = new BufferedReader(new InputStreamReader(fs.open(new Path(paths[0]))));
			String aLine;
			while ((aLine = reader.readLine()) != null) {
				Scanner s = new Scanner(aLine).useDelimiter(",");
				Point p = new Point();
				p.x = s.nextDouble();
				p.y = s.nextDouble();
				s.close();
				centroids.add(p);
			}
			
			table = new HashMap<>();
		}

		@Override
		public void map(Object key, Text value, Context context) throws IOException, InterruptedException {
			Scanner s = new Scanner(value.toString()).useDelimiter(",");
			double px = s.nextDouble();
			double py = s.nextDouble();
			s.close();
			Point p = new Point();
			p.x = px;
			p.y = py;

			Point closestCentroid = centroids.firstElement();
			double minimumDistance = Double.MAX_VALUE;
			for (Point c : centroids) {
				double existingDistance = calculateEuclideanDistance(p, c);
				if (Double.compare(existingDistance, minimumDistance) < 0) {
					minimumDistance = existingDistance;
					closestCentroid = c;
				}
			}

			if (table.containsKey(closestCentroid)) {
				// update
				Avg newAvg = new Avg(
						table.get(closestCentroid).sumX + p.x, 
						table.get(closestCentroid).sumY + p.y, 
						table.get(closestCentroid).count + 1
						);
				table.put(closestCentroid, newAvg);
			} else {
				// initialize
				table.put(closestCentroid, new Avg(p.x, p.y, 1));
			}
			
		}

		
		private double calculateEuclideanDistance(Point p, Point c) {
			double sumOfSquares = (c.y - p.y) * (c.y - p.y) + (c.x - p.x) * (c.x - p.x);
			return Math.sqrt(sumOfSquares);
		}
		
		@Override
		protected void cleanup(Context context) 
				throws IOException, InterruptedException {
			super.cleanup(context);
			for (Point c : table.keySet()) {
				System.out.println("Emitting " + c + " with value avg as " + table.get(c));
				context.write(c, table.get(c));
			}			
		}
	}

	public static class AvgReducer extends Reducer<Point, Avg, Point, Object> {
		@Override
		public void reduce(Point centroid, Iterable<Avg> avgs, Context context)
				throws IOException, InterruptedException {

			/*
			 * reduce ( c, avgs ):
					  count = 0
					  sx = sy = 0.0
					  for a in avgs
					      sx += a.sumX
					      sy += a.sumY
					      count += a.count
					  c.x = sx/count
					  c.y = sy/count
					  emit(c,null)  
  			 */
			int count = 0;
			double sumX = 0.0;
			double sumY = 0.0;
			for (Avg a : avgs) {
				sumX += a.sumX;
				sumY += a.sumY;
				count += a.count;
			}
			Point newCentroid = new Point();
			newCentroid.x = sumX / count;
			newCentroid.y = sumY / count;
			context.write(newCentroid, null);
		}
	}

	public static void main(String[] args) throws Exception {
		if (args.length != 3) {
			String errorMsg = "Invoke program with 3 parameters - the input file, the centroids file, the output directory";
			System.err.println(errorMsg);
		}

		Job job = Job.getInstance();
		job.setJarByClass(KMeans.class);
		job.setJobName("KMeans");
		// job.setOutputKeyClass(Point.class);
//		job.setOutputValueClass(null);
		job.setOutputFormatClass(TextOutputFormat.class);
		job.setMapperClass(KMeans.AvgMapper.class);
		job.setReducerClass(KMeans.AvgReducer.class);
		job.addCacheFile(new URI(args[1]));
		job.setMapOutputKeyClass(Point.class);
		job.setMapOutputValueClass(Avg.class);

		FileInputFormat.addInputPath(job, new Path(args[0]));
		FileOutputFormat.setOutputPath(job, new Path(args[2]));

		int returnValue = job.waitForCompletion(true) ? 0 : 1;
		if (job.isSuccessful()) {
			System.out.println("Job was successful! :)");
		} else if (!job.isSuccessful()) {
			System.out.println("Job was not successful. :(");
		}

		System.exit(returnValue);
	}
}
