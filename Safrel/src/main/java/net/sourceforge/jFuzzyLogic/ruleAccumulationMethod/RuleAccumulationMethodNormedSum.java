package net.sourceforge.jFuzzyLogic.ruleAccumulationMethod;


/**
 * Rule accumulation mathod: Sum
 * @author pcingola@users.sourceforge.net
 */
public class RuleAccumulationMethodNormedSum extends RuleAccumulationMethod {

	public RuleAccumulationMethodNormedSum() {
		super();
		name = "nsum";
	}

	/**
	 * @see RuleAccumulationMethod#aggregate(double, double)
	 */
	public double aggregate(double defuzzifierValue, double valueToAggregate) {
		return ( defuzzifierValue + valueToAggregate )  / Math.max(1.0, defuzzifierValue + valueToAggregate);
	}
	
	/**
	 * @see RuleAccumulationMethod#toStringFcl()
	 */
	public String toStringFcl() {
		return "ACCU : NSUM;";
	}
}
