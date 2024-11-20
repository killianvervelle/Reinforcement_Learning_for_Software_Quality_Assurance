package net.sourceforge.jFuzzyLogic.ruleAccumulationMethod;

/**
 * Rule accumulation mathod: Sum
 * @author pcingola@users.sourceforge.net
 */
public class RuleAccumulationMethodSum extends RuleAccumulationMethod {

	public RuleAccumulationMethodSum() {
		super();
		name = "sum";
	}

	/**
	 * @see RuleAccumulationMethod#aggregate(double, double)
	 */
	public double aggregate(double defuzzifierValue, double valueToAggregate) {
		return defuzzifierValue + valueToAggregate;
	}

	/**
	 * @see RuleAccumulationMethod#toStringFcl()
	 */
	public String toStringFcl() {
		return "ACCU : SUM;";
	}
}
